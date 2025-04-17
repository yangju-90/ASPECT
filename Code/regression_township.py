import glob
import os
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterstats import zonal_stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
import random
random.seed(42)
exp = 0 # whether log transform the estimators, 1: log transform, 0: no log transform
# define a dictionary for the population variables
tpop_dict = {'total_pop': 'tpop', 'age65above': 'pop65','age0_14':'pop014','age15_59':'pop1559','age60_65':'pop60'}
# define the list of covariates predicting population distribution
vars = ['mean_built', 'mean_bldg', 'mean_NTL', 'mean_rddist', 'mean_elev', 'mean_slope', 'mean_POIdensity']

# get the list of cities
cities = glob.glob(r'..\Data\city_*')
cities = [x.split('_')[-1] for x in cities]

# ############################# read data#############################
# define a dataframe to save the aggregated stats for the covariates predicting population distributions
master_output_stats = pd.DataFrame()
for i in cities:
    print('#############################Data prep on city %s#############################'%i)

    # create an output dir for each city if not exist
    dir = r'..\Processing\township\%s'%i
    if not os.path.exists(dir):
        os.makedirs(dir)

    output_dir = r'..\Output\township_output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # mask out un-inhabited areas
    print('doing masking')
    bnd = gpd.read_file(r'..\Data\city_%s\township_bnd_%s.shp' % (i, i))
    covs = glob.glob(r'..\Data\city_%s\final_*.tif' % (i))
    water = rasterio.open(r'..\Data\city_%s\final_water.tif' % (i))
    watermask = (water.read() == 1)
    builtmask = rasterio.open(r'..\Data\city_%s\final_built.tif' % (i))
    builtmask = (builtmask.read() > 0)
    bldgmask = rasterio.open(r'..\Data\city_%s\final_bldg.tif' % (i))
    bldgmask = (bldgmask.read() > 0)
    for f in [x for x in covs if 'water' not in x]:
        f_name = f.split('\\')[-1].split('.')[0][6:]
        fs = rasterio.open(f)
        kwargs = fs.meta
        kwargs.update(nodata=-999)
        fs = fs.read().astype(np.float32)
        # an inhabited area should not be water body, and should be covered by built-up area or buildings
        fs = np.where(watermask, fs, -999)
        fs = np.where(builtmask | bldgmask, fs, -999)
        out_name = r'..\Data\%s\masked_final_%s.tif' % (dir, f_name)
        with rasterio.open(out_name, 'w', **kwargs) as dst:
            dst.write(fs)
    # save a separate layer for the inhabited area
    hab = np.ones(fs.shape)
    hab = np.where(watermask, hab, -999)
    hab = np.where(builtmask | bldgmask, hab, -999)
    out_name = r'..\Data\%s\masked_final_%s.tif' % (dir, 'hab')
    with rasterio.open(out_name, 'w', **kwargs) as dst:
        dst.write(hab)

    # do zonal statistics on the covariates
    print('doing zonal stats')
    files = glob.glob(r'..\Data\%s\masked_final_*.tif' % (dir))
    output_stats = bnd.copy()
    for f in files:
        f_name = f.split('\\')[-1].split('.')[0][13:]
        stats = zonal_stats(bnd, f, copy_properties=True, stats=['mean', 'sum'])
        stats = pd.DataFrame(stats)
        stats.columns = [x + '_' + f_name for x in stats.columns]
        output_stats = pd.merge(output_stats, stats, left_index=True, right_index=True)
    # out = out[out['missing_bn'] != 9]
    for pvar in ['total_pop','age0_14','age15_59','age60_65','age65above','residents']:
        output_stats['%s_pd' % pvar] = output_stats[pvar] / output_stats['sum_hab']
    output_stats = output_stats.drop(columns='geometry')

    if len(master_output_stats) == 0:
        master_output_stats = output_stats
    else:
        master_output_stats = pd.concat([ master_output_stats, output_stats])
master_output_stats.to_csv(r'../Output/township_output/aggregated_stats.csv' )


############################# dasymetric mapping #############################
print('#############################starting the regression#############################')
perform = r'../Output/township_output/performance.csv'
p_df = pd.DataFrame() # define a dataframe to save best model parameters and performances
master_output_stats = pd.read_csv(r'../Output/township_output/aggregated_stats.csv' )
master_output_stats = master_output_stats.dropna()

idx_metrics = 0
for pvar in tpop_dict.keys():
    master_out = master_output_stats.copy()

    # filter out counties with outlying population density or inf population density
    master_out = master_out[(master_out['%s_pd' % pvar] >= 0) & (master_out['%s_pd' % pvar] < np.inf)]
    lower, upper = master_out['%s_pd' % pvar].quantile([0.005, 0.995])
    master_out = master_out[(master_out['%s_pd' % pvar] >= lower) & (master_out['%s_pd' % pvar] <= upper)]

    # whether to log transform the data
    if exp == 1:
        master_out['%s_pd' % pvar] = np.log(master_out['%s_pd' % pvar])
    else:
        master_out['%s_pd' % pvar] = master_out['%s_pd' % pvar]

    # split the counties in to training and testing sets.
    test_df = master_out.sample(frac = 0.15,random_state = 42)
    train_df = master_out.drop(test_df.index)
    X_train = train_df[vars].values
    X_test = test_df[vars].values
    y_train = train_df['%s_pd' % pvar]
    y_test = test_df['%s_pd' % pvar]


    # Initialize the Random Forest regressor
    rf = RandomForestRegressor(random_state=24)
    param_grid = {
        'n_estimators': [5, 10, 20, 40, 60, 80, 100, 150, 200, 400, 600, 800, 1000],
        'max_depth': [10, 20, 40, 50, 60, 70, 80, 90, 100]
    }

    # Perform grid search with cross-validation
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=10, verbose=1,
                               scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)

    # Get the best model from the training
    best_model = grid_search.best_estimator_
    best_score = grid_search.best_score_
    print(f'Best validation score, RMSE from training: {np.sqrt(-1*best_score):.2f}')
    print(f"Best parameters from training: {grid_search.best_params_}")

    # Evaluate the model on the test set
    y_pred = best_model.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    test_r2 =  r2_score(y_test, y_pred)
    print(f"Testing RMSE: {test_rmse:.2f}")
    print(f"Testing R2: {test_r2:.2f}")

    # save the parameters and perform of best-predicting models
    max_depth = grid_search.best_params_.get('max_depth')
    n_est = grid_search.best_params_.get('n_estimators')
    p_df.loc[idx_metrics, 'pvar'] = pvar
    p_df.loc[idx_metrics,'training_rmse']= np.sqrt(-1*best_score)
    p_df.loc[idx_metrics, 'max_depth'] = max_depth
    p_df.loc[idx_metrics, 'n_est'] = n_est
    p_df.loc[idx_metrics, 'test_rmse'] = test_rmse
    p_df.loc[idx_metrics, 'test_r2'] = test_r2
    p_df.loc[idx_metrics, 'n_samples'] = len(master_out)
    idx_metrics = idx_metrics+1
    p_df.to_csv(perform)

    # save results to the cities
    for c in cities:
        print(c)
        files = glob.glob(r'..\Processing\township\%s\masked_final_*.tif' % (c))
        idx = 0
        shape = rasterio.open(files[0]).shape
        predictors = np.empty(shape, dtype=rasterio.float64)
        for v in vars:
            fname = r'..\Processing\township\%s\masked_final_%s.tif' % (c, v.split('_')[1].split('.')[0])
            predictor = rasterio.open(fname).read().astype(np.float64).flatten()
            if idx == 0:
                predictors = predictor
            else:
                predictors  = np.vstack((predictors, predictor))
            idx = idx + 1
        predictors[np.isnan(predictors)] = 0
        weight = best_model.predict(predictors.transpose())
        weight = weight.reshape(shape)
        weight_rows, weight_cols = weight.shape
        weight = weight.reshape(1, weight_rows, weight_cols)

        # if log transform the variables, exponentiate the weights
        if exp == 1:
            weight = np.exp(weight)
        else:
            weight = weight

        # constrain population to where is building and builtup area and no water
        water = rasterio.open(r'..\Data\city_%s\final_water.tif' % (c))
        watermask = (water.read() == 1)
        builtmask = rasterio.open(r'..\Data\city_%s\final_built.tif' % (c))
        builtmask = (builtmask.read() > 0)
        bldgmask = rasterio.open(r'..\Data\city_%s\final_bldg.tif' % (c))
        bldgmask = (bldgmask.read() > 0)
        roadmask = rasterio.open(r'..\Data\city_%s\final_rddist.tif' % (c))
        roadmask = (roadmask.read() > 0)
        weight = np.where(watermask, weight, 0)
        weight = np.where(builtmask | bldgmask, weight, 0)

        # detect any negative weight
        if (np.min(weight) < 0) & (np.min(weight) != -999):
            print('warning, negative weighted detected.')

        # save the weight as a tiff file
        kwargs = water.meta
        kwargs.update(
            dtype=rasterio.float64,
            nodata=-999,
            count=1)
        out_res = r'../Processing/township/weight_%s_%s.tif' % (pvar,c)
        with rasterio.open(out_res, 'w', **kwargs) as dst:
            dst.write(weight)

        # distribute census-unit population using the weight
        df = gpd.read_file(r'..\Data\city_%s\county_bnd_%s.shp' % (c, c)) # read census unit population
        weight = rasterio.open(out_res).read().astype(np.float64)

        # get the total weight per census unit
        stats = zonal_stats(df, out_res, copy_properties=True, stats=['sum'])
        stats = pd.DataFrame(stats)
        stats = pd.merge(df, stats, left_index=True, right_index=True)
        stats = stats[['county_cod',  'sum']]

        # assign per census unit total weight to the corresponding gri cell
        zone_file = r'..\Data\city_%s\county_bnd_%s.tif' % (c, c)
        zone_file = rasterio.open(zone_file).read().astype(np.float64)
        value_dict = dict(set(zip(stats['county_cod'], stats['sum'])))
        total_weight = np.copy(zone_file)
        for old, new in value_dict.items():
            total_weight[zone_file == old] = new

        # read total population (rasterized) and distribute them to a grid cell
        # using per grid cell weight and total weight
        total_pop = rasterio.open(
            r'..\Data\city_%s\county_%s_%s.tif' % (c,
            tpop_dict.get(pvar),c)).read().astype(np.float64)  # read the total pop in raster

        pde = total_pop * weight / total_weight

        # save redistirbuted population (i.e., gridded population) to a tiff file
        out_pop = '../Output/township_output/%s_%s.tif' % (pvar,c)
        with rasterio.open(out_pop, 'w', **kwargs) as dst:
            dst.write(pde)



