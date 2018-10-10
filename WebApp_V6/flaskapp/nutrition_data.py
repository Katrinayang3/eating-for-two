import pandas as pd 
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import urllib.parse
import seaborn
from scipy.spatial.distance import pdist, squareform
from sklearn.externals import joblib
import spacy

def optimal_recipe(recipename):   # for 1 
    db = pd.read_csv('app_data/BI_onerecipe_v4.csv')

    dailyvals = {'Total Fat': 65, 'Saturated Fat': 20, 'Cholesterol': 300, 'Sodium': 2400, 'Potassium': 3500,
             'Total Carbohydrates': 300, 'Dietary Fiber': 25, 'Protein': 100, 'Sugars': 31.5,'Vitamin A': 8000,
             'Vitamin C': 60, 'Calcium': 1300, 'Iron': 27, 'Thiamin': 1.7, 'Niacin': 20, 'Vitamin B6': 2.5, 'Folate': 800,
            'Magnesium': 400, 'Energy': 2300}
    recipe_input = db[db['recipename']==recipename]

    # construct a set of combined recipes
    db.iloc[:,3:] += recipe_input.iloc[0,3:]
    
#     db.set_index(['recipename','labels'], inplace=True)
    db_combine = db[['recipename','labels','Total Fat','Saturated Fat','Cholesterol','Sodium', 'Total Carbohydrates','Sugars', \
                                   'Protein','Dietary Fiber','Vitamin A','Vitamin C','Calcium','Potassium','Iron',\
                                   'Thiamin','Niacin','Vitamin B6','Magnesium','Folate', 'Calories']]

   
    db_combine['Carb_std'] = db_combine['Total Carbohydrates'] / db_combine['Calories'] * dailyvals['Energy']
    db_combine['Protein_std'] = db_combine['Protein'] / db_combine['Calories'] * dailyvals['Energy']
    db_combine['VitA_std'] = db_combine['Vitamin A'] / db_combine['Calories'] * dailyvals['Energy']
    db_combine['VitC_std'] = db_combine['Vitamin C'] / db_combine['Calories'] * dailyvals['Energy']
    db_combine['Calcium_std'] = db_combine['Calcium'] / db_combine['Calories'] * dailyvals['Energy']
    db_combine['Iron_std'] = db_combine['Iron'] / db_combine['Calories'] * dailyvals['Energy']
    db_combine['Thiamin_std'] = db_combine['Thiamin'] / db_combine['Calories'] * dailyvals['Energy']
    db_combine['Niacin_std'] = db_combine['Niacin'] / db_combine['Calories'] * dailyvals['Energy']
    db_combine['VitB6_std'] = db_combine['Vitamin B6'] / db_combine['Calories'] * dailyvals['Energy']
    db_combine['Magnesium_std'] = db_combine['Magnesium'] / db_combine['Calories'] * dailyvals['Energy']
    db_combine['Folate_std'] = db_combine['Folate'] / db_combine['Calories'] * dailyvals['Energy']
    db_combine['Potassium_std'] = db_combine['Potassium'] / db_combine['Calories'] * dailyvals['Energy']
    db_combine['Dietary Fiber_std'] = db_combine['Dietary Fiber'] / db_combine['Calories'] * dailyvals['Energy']
    # moderate nutrient
    db_combine['Total Fat_std'] = db_combine['Total Fat'] / db_combine['Calories'] * dailyvals['Energy']
    db_combine['Saturated Fat_std'] = db_combine['Saturated Fat'] / db_combine['Calories'] * dailyvals['Energy']
    db_combine['Cholesterol_std'] = db_combine['Cholesterol'] / db_combine['Calories'] * dailyvals['Energy']
    db_combine['Sodium_std'] = db_combine['Sodium'] / db_combine['Calories'] * dailyvals['Energy']
    db_combine['Sugars_std'] = db_combine['Sugars'] / db_combine['Calories'] * dailyvals['Energy']

    # calculate the BI
    db_combine['Carb_BI'] = [x / dailyvals['Total Carbohydrates'] * 5 if x < dailyvals['Total Carbohydrates'] else 5 for x in db_combine['Carb_std']]
    db_combine['Protein_BI'] = [x / dailyvals['Protein'] * 15 if x < dailyvals['Protein'] else 15 for x in db_combine['Protein_std']]
    db_combine['VitA_BI'] = [x / dailyvals['Vitamin A'] * 5 if x < dailyvals['Vitamin A'] else 5 for x in db_combine['VitA_std']]
    db_combine['VitC_BI'] = [x / dailyvals['Vitamin C'] * 5 if x < dailyvals['Vitamin C'] else 5 for x in db_combine['VitC_std']]
    db_combine['Calcium_BI'] = [x / dailyvals['Calcium'] * 5 if x < dailyvals['Calcium'] else 5 for x in db_combine['Calcium_std']]
    db_combine['Iron_BI'] = [x / dailyvals['Iron'] * 5 if x < dailyvals['Iron'] else 5 for x in db_combine['Iron_std']]
    db_combine['Thiamin_BI'] = [x / dailyvals['Thiamin'] * 5 if x < dailyvals['Thiamin'] else 5 for x in db_combine['Thiamin_std']]
    db_combine['Niacin_BI'] = [x / dailyvals['Niacin'] * 5 if x < dailyvals['Niacin'] else 5 for x in db_combine['Niacin_std']]
    db_combine['VitB6_BI'] = [x / dailyvals['Vitamin B6'] * 5 if x < dailyvals['Vitamin B6'] else 5 for x in db_combine['VitB6_std']]
    db_combine['Magnesium_BI'] = [x / dailyvals['Magnesium'] * 5 if x < dailyvals['Magnesium'] else 5 for x in db_combine['Magnesium_std']]
    db_combine['Folate_BI'] = [x / dailyvals['Folate'] * 5 if x < dailyvals['Folate'] else 5 for x in db_combine['Folate_std']]
    db_combine['Potassium_BI'] = [x / dailyvals['Potassium'] * 5 if x < dailyvals['Potassium'] else 5 for x in db_combine['Potassium_std']]
    db_combine['Dietary Fiber_BI'] = [x / dailyvals['Dietary Fiber'] * 5 if x < dailyvals['Dietary Fiber'] else 5 for x in db_combine['Dietary Fiber_std']]

    # nutrient of moderate intake
    db_combine['Total Fat_BI'] = [(dailyvals['Total Fat'] - x) / dailyvals['Total Fat'] * 5 if x < dailyvals['Total Fat'] else 0 for x in db_combine['Total Fat_std'] ]
    db_combine['Saturated Fat_BI'] = [(dailyvals['Saturated Fat'] - x) / dailyvals['Saturated Fat'] * 5 if x < dailyvals['Saturated Fat'] else 0 for x in db_combine['Saturated Fat_std'] ]
    db_combine['Cholesterol_BI'] = [(dailyvals['Cholesterol'] - x) / dailyvals['Cholesterol'] * 5 if x < dailyvals['Cholesterol'] else 0 for x in db_combine['Cholesterol_std'] ]
    db_combine['Sodium_BI'] = [(dailyvals['Sodium'] - x) / dailyvals['Sodium'] * 5 if x < dailyvals['Sodium'] else 0 for x in db_combine['Sodium_std'] ]
    db_combine['Sugars_BI'] = [(dailyvals['Sugars'] - x) / dailyvals['Sugars'] * 5 if x < dailyvals['Sugars'] else 0 for x in db_combine['Sugars_std'] ]


    # total BI
    db_combine['BI_combine'] = db_combine['Carb_BI'] + db_combine['Protein_BI'] + db_combine['VitA_BI'] + db_combine['VitC_BI'] + db_combine['Calcium_BI'] + db_combine['Iron_BI'] + \
                db_combine['Thiamin_BI'] + db_combine['Niacin_BI'] + db_combine['VitB6_BI'] + db_combine['Magnesium_BI'] + db_combine['Folate_BI'] + db_combine['Potassium_BI'] + \
                db_combine['Dietary Fiber_BI'] + db_combine['Total Fat_BI'] + db_combine['Saturated Fat_BI'] + db_combine['Sodium_BI'] + db_combine['Sugars_BI']

    db_combine.to_csv('app_data/BI_combined_v4.csv')

    db_recommend_recipe = db_combine['recipename'][db_combine['BI_combine'].idxmax()]
    db_recommend_BI_score = db_combine['BI_combine'][db_combine['BI_combine'].idxmax()]
                                     
    # db_recommend_recipe: is the name of the recipe with the highest BI Score
    # db_recommend_BI_score: the highest BI score
    # db_combine: a new dataframe with all BI score for combined recipes
    # return db_recommend_recipe, db_recommend_BI_score, db_combine
    return db_recommend_recipe, db_recommend_BI_score

def local_recipe(recipename, similarity):   # for 1 input
    db = pd.read_csv('app_data/BI_combined_v4.csv')
    dist_matrix_transpose = np.loadtxt('app_data/dist_matrix_transpose.csv', delimiter=",")

    recipe_input = db[db['recipename']==recipename]

    
    cluster_index = recipe_input['labels']
    similarity = np.max(dist_matrix_transpose, axis=0)[cluster_index] * similarity  # similarity is scaled by the distance matrix
    a = [row[cluster_index] for row in dist_matrix_transpose]
    cluster_option = min(enumerate(a), key=lambda x: abs(x[1]-similarity))  # get the cluster that is most close to the chosen similarity
    db_sub = db[db['labels']==cluster_option[0]]
    db_sub_sorted = db_sub.sort_values(['BI_combine'], ascending = 0)
    
    return db_sub_sorted.head(5)     # return the 5 recipe with the top BI score within the selected cluster

def predict(recipename):
	'''use the saved GMM model to predict class label for selected recipe'''
	#load recipe db that has been sqrt transformed
	db = pd.read_csv('app_data/db_nutr_sqrt.csv').set_index('recipename')
	db = db.iloc[:,:9]

	#load standard scale, pca, gmm used during training
	pca = joblib.load('app_data/distpca.pkl')
	standardscale = joblib.load('app_data/standardscaler.pkl')
	mdl = joblib.load('app_data/gmm4.pkl')

	nutrfacts = db.loc[recipename]
	nutrfacts = np.array(nutrfacts,ndmin=2)
	nutrfacts = standardscale.transform(nutrfacts)

	db_transformed = standardscale.transform(db.values) 
	db_dist = squareform(pdist(np.append(db_transformed,nutrfacts,axis=0)))
	label = mdl.predict(pca.transform(np.array(db_dist[-1,:-1],ndmin=2)))
	
	return label

def recommendations(classlabel,recipename):
	'''given class label of specific recipe and recipe name, return list of recipes in the 
	same, healthier, and even healthier groups, ranked by ingredient similarity'''
	
	#names of recipes in a simnilar, better, or even better health class
	similar,better,best = betterclasses(classlabel)

	#load word vector model used to filter recipes by ingred similarity
	nlp = spacy.load('app_data/recipe_ingred_word2vec_lg')

	#load ingred db (cleaned)
	ingredsdb = pd.read_csv('app_data/ingreds_db_cleaned.csv').rename(columns={'Unnamed: 0':'recipename'})
	ingredsdb = ingredsdb.set_index('recipename')
	
	#load recipe db 
	recipedb = pd.read_csv('app_data/allrecipes_nutr_labels.csv').set_index('recipename')

	#get similarity scores to target recipe
	targetingred = nlp(ingredsdb.loc[recipename].ingredients)
	similarity = []
	for num in range(0,len(similar)):
		similarity.append(targetingred.similarity(nlp(ingredsdb.loc[similar[num]].ingredients)))
	similarity = np.array(similarity)
	similar = similar[(-similarity).argsort()[:10]]
	similarity = similarity[(-similarity).argsort()[:10]]
	similar_recipes = recipedb.loc[similar]
	#similar_recipes['ingred_similarity'] = similarity

	if len(better) > 0:
		similarity = []
		for num in range(0,len(better)):
			similarity.append(targetingred.similarity(nlp(ingredsdb.loc[better[num]].ingredients)))
		similarity = np.array(similarity)
		better = better[(-similarity).argsort()[:10]]
		similarity = similarity[(-similarity).argsort()[:10]]
		better_recipes = recipedb.loc[better]
	#	better_recipes['ingred_similarity'] = similarity
	else:
		better_recipes = recipedb.loc[better]

	if len(best) > 0:
		similarity = []
		for num in range(0,len(best)):
			similarity.append(targetingred.similarity(nlp(ingredsdb.loc[best[num]].ingredients)))
		similarity = np.array(similarity)
		best = best[(-similarity).argsort()[:10]]
		similarity = similarity[(-similarity).argsort()[:10]]
		best_recipes = recipedb.loc[best]
	#	best_recipes['ingred_similarity'] = similarity
	else:
		best_recipes = recipedb.loc[best]

	return similar_recipes,better_recipes, best_recipes

def plot_bi_comparison(recipename, recommended_recipe):
    # get the BI score for the input recipe
    db = pd.read_csv('app_data/BI_onerecipe_v4.csv')
    recipe_input = db[db['recipename']==recipename]
    recipe_input = recipe_input[['recipename', 'Total Fat_BI','Saturated Fat_BI','Cholesterol_BI','Sodium_BI', 'Carb_BI','Sugars_BI',\
                                      'Protein_BI','Dietary Fiber_BI','VitA_BI','VitC_BI','Calcium_BI','Iron_BI','Potassium_BI','Thiamin_BI', \
                                       'Niacin_BI','VitB6_BI', 'Magnesium_BI','Folate_BI','BI']]

    # get the BI score for the best combined recipes (recommended in part 2)
    db_combine = pd.read_csv('app_data/BI_combined_v4.csv')
    recipe_recomend = db_combine[db_combine['recipename'] == recommended_recipe]
    recipe_recomend = recipe_recomend.rename(columns = {'BI_combine':'BI'})
    recipe_recomend = recipe_recomend[['recipename', 'Total Fat_BI','Saturated Fat_BI','Cholesterol_BI','Sodium_BI', 'Carb_BI','Sugars_BI',\
                                      'Protein_BI','Dietary Fiber_BI','VitA_BI','VitC_BI','Calcium_BI','Iron_BI','Potassium_BI','Thiamin_BI',\
                                       'Niacin_BI','VitB6_BI', 'Magnesium_BI','Folate_BI','BI']]

    # rename the recipename for better understanding of the plot
    # recipe_input.at[0,'recipename'] = 'Lunch'
    # recipe_recomend.at[0,'recipename'] = 'Lunch + Dinner'
    recipe_input['recipename'] = 'Lunch'
    recipe_recomend['recipename'] = 'Lunch + Dinner'

    # get a new dataframe of 2 recipes for plot
    recipe_two = recipe_input.append(recipe_recomend, ignore_index=True).set_index('recipename')


#     recipe_two.to_csv('Data/recipe_two.csv', index = False)

#     recipe_two = pd.read_csv('Data/recipe_two.csv').set_index('recipename')

    #combine nutr info and change df to long form 	
    combined_nutrfacts = recipe_two.T.reindex(index=['Total Fat_BI','Saturated Fat_BI','Cholesterol_BI','Sodium_BI', 'Carb_BI','Sugars_BI',\
                                  'Protein_BI','Dietary Fiber_BI','VitA_BI','VitC_BI','Calcium_BI','Iron_BI','Potassium_BI','Thiamin_BI',\
                                   'Niacin_BI','VitB6_BI', 'Magnesium_BI','Folate_BI'])
    combined_nutrfacts = combined_nutrfacts.reset_index().melt(id_vars=['index'])
    combined_nutrfacts = combined_nutrfacts.rename(columns={'index':'nutrients'})


    img = io.BytesIO()
    plt.figure(figsize=(10,15))
    g = seaborn.barplot(y="nutrients",x="value",hue="recipename",data=combined_nutrfacts,
        palette=['k','g'],alpha=0.4)
    ax=plt.gca()
    plt.xlabel('BI score',fontsize=20)
    plt.ylabel('')
    #ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()

    ax.spines['top'].set_linewidth(3)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_linewidth(3)
    # ax.axvline(15,lw=4,ls=':')
    # ax.axvline(5,lw=4,ls=':')
    ax.tick_params(labelsize=20)
    ax.set_yticklabels(labels=combined_nutrfacts['nutrients'])
    ax.legend(loc=1,frameon=False,fontsize=15)


    [i.set_color('r') for i in ax.get_yticklabels() if i.get_text() in ['Total Fat_BI','Saturated Fat_BI',
        'Cholesterol_BI','Sodium_BI','Carb_BI','Sugars_BI']]
    [i.set_color('g') for i in ax.get_yticklabels() if i.get_text() in ['Potassium_BI','Dietary Fiber_BI',
        'Protein_BI','VitA_BI','VitC_BI','Calcium_BI','Iron_BI']]

    plt.tight_layout()
    plt.savefig(img,format='png')
    img.seek(0)
    plot_url = urllib.parse.quote(base64.b64encode(img.read()).decode())
    
    return plot_url

