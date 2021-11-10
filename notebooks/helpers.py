from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_validate
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, make_scorer
from scipy.stats import uniform, randint
from eli5.sklearn import PermutationImportance
import eli5
# from sklearn.model_selection import 
import numpy as np
import pandas as pd
# import re
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import shap

def adj_rSquared_func(y, y_pred, n, p):
    r2 = r2_score(y, y_pred)
    return 1 - ((1 - r2) * ((n - 1) / (n - p - 1)))

def getR2AndCVResults(path, X, y, X_test, y_test, numObservations = 779, numFeatures = 109):
    file = open(path, 'rb')
    model = pickle.load(file)
    file.close()

    if path == '../models/XGB_ALL_PARAMS':
        model = model.best_estimator_

    y_pred = model.predict(X_test)

    scoring_dict = {'adj_rSquared': make_scorer(adj_rSquared_func, n = numObservations, p = numFeatures), 
                    'rSquared': 'r2', 
                    'mae': 'neg_mean_absolute_error'}

    r2_base = r2_score(y_test, y_pred)
    adj_r2_base = adj_rSquared_func(y_test, y_pred, n = numObservations, p = numFeatures)

    params = model.get_params()

    if path.find('../models/RF') != -1:
        new_model = RandomForestRegressor(random_state=0, n_jobs = -1, criterion = 'mae')
    else:
        new_model = buildNewModel(params)
    
    cv_results = cross_validate(new_model, X, y, cv = 5, scoring = scoring_dict)

    return cv_results, r2_base, adj_r2_base

def shapPlots(pathToModel, title, test_data):
    model_file = open(pathToModel, 'rb')
    model = pickle.load(model_file)
    model_file.close()

    explainer = shap.Explainer(model)
    shap_values = explainer(test_data)

    shap.plots.beeswarm(shap_values, max_display = 20, show = False)
    plt.title(title, size = 20)
    plt.savefig('../summary/figures/shap/' + title + ' SHAP Beeswarm Plot')
    plt.show()

    shap.plots.bar(shap_values, max_display = 20, show = False)
    plt.title(title, size = 20)
    plt.savefig('../summary/figures/shap/' + title + ' SHAP Bar Plot')
    plt.show()

# def shapAnalysis():

def writeModelToFile(name, model):
    file = open(name, 'wb')
    pickle.dump(model, file)
    file.close()

def dumpModels(xgb_model, xgb_corr, xgb_RF_perm, xgb_XGB_perm, regr, regr_corr, regr_RF_perm, regr_XGB_perm):
    writeModelToFile('../models/XGB_ALL_PARAMS', xgb_model)
    writeModelToFile('../models/XGB_TOP20_CORR_PARAMS', xgb_corr)
    writeModelToFile('../models/XGB_TOP20_RF_PERM_PARAMS', xgb_RF_perm)
    writeModelToFile('../models/XGB_TOP20_XGB_PERM_PARAMS', xgb_XGB_perm)
    writeModelToFile('../models/RF_ALL_PARAMS', regr)
    writeModelToFile('../models/RF_TOP20_CORR_PARAMS', regr_corr)
    writeModelToFile('../models/RF_TOP20_RF_PERM_PARAMS', regr_RF_perm)
    writeModelToFile('../models/RF_TOP20_XGB_PERM_PARAMS', regr_XGB_perm)

def writeToFile(mae, mae_corr, mae_RF_perm, mae_XGB_perm, X_corr, perm_regr, perm_xgb):
    mae = mae.append(mae_corr)
    mae = mae.append(mae_RF_perm)
    mae = mae.append(mae_XGB_perm)
    mae.index = ['all_tf', 't20_corr', 't20_RF_perm', 't20_XGB_perm']
    mae.to_csv('../summary/analysis/CD46_MAE.csv')
    X_corr.to_csv('../summary/analysis/CD46_corr.csv')
    perm_regr.to_csv('../summary/analysis/CD46_RF_perm.csv')
    perm_xgb.to_csv('../summary/analysis/CD46_XGB_perm.csv')

def display_scores(scores):
    print("Scores: {0}\nMean: {1:.3f}\nStd: {2:.3f}".format(scores, np.mean(scores), np.std(scores)))
    
def report_best_scores(results, n_top=3):
    for i in range(1, n_top + 1):
#         i = 201 - i
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

def permImportance(params, X_train, X_test, y_train, y_test, n = 20):
    regr = RandomForestRegressor(random_state=0, n_jobs = -1, criterion = 'mae')

    regr.fit(X_train, y_train)

    xgb_model = buildNewModelAndFit(params, X_train, y_train)

    perm = PermutationImportance(xgb_model, random_state=1).fit(X_test, y_test)

    perm_xgb = eli5.formatters.as_dataframe.explain_weights_df(perm, feature_names = X_test.columns.tolist())
    
    perm = PermutationImportance(regr, random_state=1).fit(X_test, y_test)

    perm_regr = eli5.formatters.as_dataframe.explain_weights_df(perm, feature_names = X_test.columns.tolist())

    X_RF_train = X_train[perm_regr.loc[:19, 'feature']]
    X_RF_test = X_test[perm_regr.loc[:19, 'feature']]

    X_XGB_train = X_train[perm_xgb.loc[:19, 'feature']]
    X_XGB_test = X_test[perm_xgb.loc[:19, 'feature']]

    return perm_xgb, perm_regr, X_RF_train, X_RF_test, X_XGB_train, X_XGB_test
# perm_xgb.to_csv('CD38_XGB_perm.csv')

    # xgb_model = xgb.XGBRegressor(n_jobs = -1, random_state = 42, colsample_bylevel = 0.5658140364286011, 
    #                             colsample_bynode = 0.8078536842438382, colsample_bytree = 0.7051418651679482, 
    #                             gamma = 0.7909390660598336, learning_rate = 0.0968001207373439, 
    #                             max_depth = 2, n_estimators = 183, subsample = 0.8963379482245502)

    # xgb_model.fit(X_log_train, y_log_train, eval_metric = 'mae')

def get_params(results, rank = 1):
    candidates = np.flatnonzero(results['rank_test_score'] == rank)
    return results['params'][candidates[0]]

def buildNewModelAndFit(params, X_train, y_train, nJobs = -1, randomState = 42):
    xgb_model = xgb.XGBRegressor(n_jobs = nJobs, random_state = randomState, colsample_bylevel = params['colsample_bylevel'], 
                            colsample_bynode = params['colsample_bynode'], colsample_bytree = params['colsample_bytree'], 
                            gamma = params['gamma'], learning_rate = params['learning_rate'], 
                            max_depth = params['max_depth'], n_estimators = params['n_estimators'], subsample = params['subsample'])
    xgb_model.fit(X_train, y_train, eval_metric = 'mae')
    return xgb_model

def buildNewModel(params, nJobs = -1, randomState = 42):
    xgb_model = xgb.XGBRegressor(n_jobs = nJobs, random_state = randomState, colsample_bylevel = params['colsample_bylevel'], 
                            colsample_bynode = params['colsample_bynode'], colsample_bytree = params['colsample_bytree'], 
                            gamma = params['gamma'], learning_rate = params['learning_rate'], 
                            max_depth = params['max_depth'], n_estimators = params['n_estimators'], subsample = params['subsample'])
    # xgb_model.fit(X_train, y_train, eval_metric = 'mae')
    return xgb_model

def exploratoryPlots(y, X):
    plt.figure(figsize=(15,8))
    sns.distplot(y)
    plt.title("CD46 expression")
    plt.savefig('../summary/figures/CD46_expression_distribution_patients.png')
    plt.figure(figsize=(19,25))
    sns.barplot(data=X, orient = 'h')
    plt.savefig('../summary/figures/TF_expression_patients.png')

def getMAEandPlots(xgb, X_train, X_test, y_train, y_test, title = 'All Features'):
    regr = RandomForestRegressor(random_state=0, n_jobs = -1, criterion = 'mae')
    regr.fit(X_train, y_train)
    y_pred_regr = regr.predict(X_test)
    y_pred_regr = pd.Series(y_pred_regr, index = y_test.index)
    y_pred_xgb = xgb.predict(X_test)
    y_pred_xgb = pd.Series(y_pred_xgb, index = y_test.index)

    mae = pd.DataFrame({'xgb': [mean_absolute_error(y_test, y_pred_xgb)], 
                         'rf': [mean_absolute_error(y_test, y_pred_regr)]})

    y_pred_regr = pd.DataFrame(y_pred_regr)
    y_pred_regr.columns = ['CD46predicted']

    y_pred_regr['CD46'] = y_test
    y_pred_regr['Type'] = 'Testing Data RF'

    y_pred_regr_train = regr.predict(X_train)

    y_pred_regr_train = pd.Series(y_pred_regr_train, index = y_train.index)

    y_pred_regr_train = pd.DataFrame(y_pred_regr_train)
    y_pred_regr_train.columns = ['CD46predicted']

    y_pred_regr_train['CD46'] = y_train
    y_pred_regr_train['Type'] = 'Training Data RF'
    y_pred_regr_comb = pd.concat([y_pred_regr, y_pred_regr_train])

    y_pred_xgb = pd.DataFrame(y_pred_xgb)
    y_pred_xgb.columns = ['CD46predicted']

    y_pred_xgb['CD46'] = y_test
    y_pred_xgb['Type'] = 'Testing Data XGB'

    y_pred_xgb_train = xgb.predict(X_train)

    y_pred_xgb_train = pd.Series(y_pred_xgb_train, index = y_train.index)

    y_pred_xgb_train = pd.DataFrame(y_pred_xgb_train)
    y_pred_xgb_train.columns = ['CD46predicted']

    y_pred_xgb_train['CD46'] = y_train
    y_pred_xgb_train['Type'] = 'Training Data XGB'
    y_pred_xgb_comb = pd.concat([y_pred_xgb, y_pred_xgb_train])

    y_pred_comb = pd.concat([y_pred_xgb_comb, y_pred_regr_comb])

    plt.figure(figsize=(20,20))

    sns.lmplot(x = 'CD46', y = 'CD46predicted', hue = 'Type', data = y_pred_regr_comb, height = 10, legend = False)

    plt.xlabel('Actual', size = 16)

    plt.ylabel('Predicted', size = 16)

    plt.ylim(9.5, 17)
    plt.xlim(9.5, 17)

    plt.legend(loc = 'lower right', fontsize = 14)

    plt.title('RF using ' + title, size = 20)
    
    plt.savefig('../summary/figures/RF using ' + title + '.png', bbox_inches='tight')
    
    plt.show()

    sns.lmplot(x = 'CD46', y = 'CD46predicted', hue = 'Type', data = y_pred_xgb_comb, height = 10, legend = False)

    plt.xlabel('Actual', size = 16)

    plt.ylabel('Predicted', size = 16)

    plt.ylim(9.5, 17)
    plt.xlim(9.5, 17)

    plt.legend(loc = 'lower right', fontsize = 14)

    plt.title('XGB tuned model using ' + title, size = 20)

    plt.savefig('../summary/figures/XGB tuned model using ' + title + '.png', bbox_inches='tight')

    plt.show()

    sns.lmplot(x = 'CD46', y = 'CD46predicted', hue = 'Type', data = y_pred_comb, height = 10, legend = False)

    plt.xlabel('Actual', size = 16)

    plt.ylabel('Predicted', size = 16)

    plt.ylim(9.5, 17)
    plt.xlim(9.5, 17)

    plt.legend(loc = 'lower right', fontsize = 14)

    plt.title('RF and XGB using ' + title, size = 20)

    plt.savefig('../summary/figures/RF and XGB using ' + title + '.png', bbox_inches='tight')

    plt.show()

    return mae, regr

def getModelAndBestParams(X, y, n = 50):
    params = {
        "colsample_bytree": uniform(0.5, 0.5),
        "colsample_bylevel": uniform(0.5, 0.5),
        "colsample_bynode": uniform(0.5, 0.5),
        "gamma": uniform(0, 1),
        "learning_rate": uniform(0, 1), # default 0.1 
        "max_depth": randint(2, 10), # default 3
        "n_estimators": randint(100, 200), # default 100
        "subsample": uniform(0.5, 0.5)
    }
    xgb_model = xgb.XGBRegressor(n_jobs = -1, random_state = 42)
    search = RandomizedSearchCV(xgb_model, param_distributions=params, random_state=42, n_iter=n, refit= True, 
                                scoring = "neg_mean_absolute_error", n_jobs=-1, return_train_score=True)
    search.fit(X, y)
    return search, get_params(search.cv_results_)

def getProcessedData(goi_id, testSize = 0.2, randomState = 42):
    pat = pd.read_csv('../data/MMRF_CoMMpass_IA13a_E74GTF_HtSeq_Gene_Counts.csv')
    pat = pat.set_index(pat.GENE_ID)
    pat = pat.drop(columns = 'GENE_ID')
    pat = pat.loc[:, pat.columns.str.endswith('1_BM') | pat.columns.str.endswith('1_PB')]

    # cell = pd.read_csv('HMCL66_HTSeq_GENE_Counts_v2.csv')
    # cell = cell.set_index(cell.Sample)
    # cell = cell.drop(['Sample', 'GENE_NAME'], axis = 1)

    # ccle = pd.read_csv('CCLE_RNAseq_genes_counts_20180929.csv')
    # ccle['Name'] = ccle.Name.map(lambda x : re.sub('\.\d+$', '', x))
    # ccle = ccle.set_index(ccle.Name)
    # ccle = ccle.drop(['Name', 'Description'], axis = 1)
    # ccle_hlt = ccle.loc[:,ccle.columns.map(lambda x : x.endswith('HAEMATOPOIETIC_AND_LYMPHOID_TISSUE'))]

    goi_names = pd.read_csv('../data/cd46genes.csv', header = None)
    goi_names = goi_names.rename({0 :'GENE_NAMES'}, axis = 1)

    goi_pat = pat.loc[goi_id]
    goi_pat = goi_pat.set_index(goi_names.iloc[:, 0])
    goi_pat = goi_pat.T
    goi_pat_log = np.log2(goi_pat)
    goi_pat_log = goi_pat_log.replace(-np.inf, 0)

    y_log = goi_pat_log.CD46
    X_log = goi_pat_log.drop('CD46', axis = 1)

    X_log_train, X_log_test, y_log_train, y_log_test = train_test_split(X_log, y_log, test_size=testSize, random_state=randomState)
    
    return goi_pat_log, X_log, y_log, X_log_train, X_log_test, y_log_train, y_log_test

def getCorrAndHighCorrFeatures(X, y, X_train, X_test, n = 20):
    X_corr = X.corrwith(y, method = 'spearman').abs().sort_values(ascending = False)
    X_train_corr = X_train[X_corr[:n].index]
    X_test_corr = X_test[X_corr[:n].index]
    return X_corr, X_train_corr, X_test_corr

def getGOI():
    goi_id = ['ENSG00000172216', #CEBPB works
            'ENSG00000101412', # E2F1 works
            'ENSG00000164330', # EBF1 works
            'ENSG00000126767', # ELK1 works
            'ENSG00000134954', # ETS1 works
            'ENSG00000075426', # FOSL2 works
            'ENSG00000129514', # FOXA1 works
            'ENSG00000102145', # GATA1 works
            'ENSG00000125347', # IRF1 works
            'ENSG00000130522', # JUND works
            'ENSG00000001167', # NFYA works
            'ENSG00000120837', # NFYB works
            'ENSG00000196092', # PAX5 works
            'ENSG00000173039', # RELA works
            'ENSG00000186350', # RXRA works
            'ENSG00000185591', # SP1 works
            'ENSG00000147133', # TAF1 works
            'ENSG00000178913', # TAF7 works
            'ENSG00000100811', # YY1 works
            'ENSG00000123268', # ATF1 works
            'ENSG00000162772', # ATF3 works
            'ENSG00000134107', # BHLHE40 works
            'ENSG00000082258', # CCNT2 works
            'ENSG00000153922', # CHD1 works
            'ENSG00000173575', # CHD2 works
            'ENSG00000118260', # CREB1 works
            'ENSG00000169016', # E2F6 works
            'ENSG00000120738', # EGR1 works
            'ENSG00000158711', # ELK4 works
            'ENSG00000125798', # FOXA2 works
            'ENSG00000154727', # GABPA works
            'ENSG00000116478', # HDAC1 works
            'ENSG00000196591', # HDAC2 works
            'ENSG00000101076', # HNF4A works
            'ENSG00000117139', # KDM5B works
            'ENSG00000073614', # KDM5A works
            'ENSG00000125952', # MAX works
            'ENSG00000103495', # MAZ works
            'ENSG00000119950', # MXI1 works
            'ENSG00000101057', # MYBL2 works
            'ENSG00000136997', # MYC works
            'ENSG00000185551', # NR2F2 works
            'ENSG00000140464', # PML works
            'ENSG00000117222', # RBBP5 works
            'ENSG00000084093', # REST works
            'ENSG00000143390', # RFX5 works
            'ENSG00000169375', # SIN3A works
            'ENSG00000066336', # SPI1 works
            'ENSG00000072310', # SREBP1 works
            'ENSG00000126561', # STAT5A works
            'ENSG00000162367', # TAL1 works
            'ENSG00000112592', # TBP works
            'ENSG00000140262', # TCF12 works
            'ENSG00000071564', # TCF3 works
            'ENSG00000131931', # THAP1 works
            'ENSG00000158773', # USF1 works
            'ENSG00000105698', # USF2 works
            'ENSG00000169083', # AR works
            'ENSG00000143437', # ARNT works
            'ENSG00000245848', # CEBPA works
            'ENSG00000141905', # CTF works
            'ENSG00000091831', # ESR1 works
            'ENSG00000157557', # ETS2 works
            'ENSG00000175832', # ETV4 works
            'ENSG00000049768', # FOXP3 works
            'ENSG00000179348', # GATA2 works
            'ENSG00000005436', # GCFC2 works
            'ENSG00000135100', # HNF1A works
            'ENSG00000128710', # HOXD10 works
            'ENSG00000128709', # HOXD9 works
            'ENSG00000185811', # IKZF1 works
            'ENSG00000138795', # LEF1 works
            'ENSG00000118513', # MYB works
            'ENSG00000196712', # NF1 works
            'ENSG00000131196', # NFATC1 works
            'ENSG00000101096', # NFATC2 works
            'ENSG00000109320', # NFKB1 works
            'ENSG00000175745', # NR2F1 works
            'ENSG00000113580', # NR3C1 works
            'ENSG00000082175', # PGR works
            'ENSG00000186951', # PPARA works
            'ENSG00000131759', # RARA works
            'ENSG00000077092', # RARB works
            'ENSG00000184895', # SRY works
            'ENSG00000115415', # STAT1 works
            'ENSG00000138378', # STAT4 works
            'ENSG00000196628', # TCF4 works
            'ENSG00000148737', # TCF7L2 works
            'ENSG00000137203', # TFAP2A works
            'ENSG00000141510', # TP53 works
            'ENSG00000111424', # VDR works
            'ENSG00000100219', # XBP1 works
            'ENSG00000074219', # TEAD2 works
            'ENSG00000117335' # CD46 works
            ]
    
    return goi_id