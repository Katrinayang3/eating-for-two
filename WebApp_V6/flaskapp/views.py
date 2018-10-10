from flaskapp import app
from flask import render_template, request, Flask
import pandas as pd 
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import urllib.parse
import flaskapp.nutrition_data as nd 
import re

#load stored allrecipes databases
recipedb = pd.read_csv('app_data/BI_onerecipe_v4.csv')
recipedb = recipedb.set_index('recipename')
optimal_recipe_bi = ''
usr_recipe_bi = ''
usr_recipe = ''
#  optimal_recipe = ''
similarity = 0.0
allergy = []

@app.route('/')
# @app.route('/index')
@app.route('/index', methods=['GET', 'POST'])
def index():	
	'''Choose random sample of 10 recipes from stored database and display names on sidebar
	Saves db indices for the random sample so that list is maintained when switching to 
	second page'''
	maindishlist = []
	indices = []
	global similarity, allergy
	if request.method == 'POST':
		for key, value in request.form.items():
			if key == 'similarity':
				similarity = float(value)
			if key == 'allergy' and value != '':
				allergy.append(value)
		print(similarity)
		print(allergy)
	#similarity = float(request.args.get('similarity'))	
	for num in np.random.choice(len(recipedb),size=10,replace=False):
		maindishlist.append(recipedb.iloc[num].name)
		indices.append(num)
	np.save('app_data/maindish_indices', indices)
	return render_template("index.html", 
				maindishes = maindishlist, 
				similarity = similarity)


@app.route('/firstchoice', methods=['GET', 'POST'])
def firstchoice_displfacts():
	'''Display nutritional facts for selected main dish recipe, show list of recommended alternative recipes'''
	global similarity, allergy
	if request.method == 'POST':
	#if request.method == 'GET':
		for key, value in request.form.items():
			if key == 'similarity':
				similarity = float(value)
			if key == 'allergy'and value != '':
				allergy.append(value)
		print(similarity)
		print(allergy)
	#similarity = float(request.args.get('similarity'))	
	#indices for sample of 10 recipes from db from /index for the sidebar 
	maindish_indices = np.load('app_data/maindish_indices.npy')
	maindishlist = recipedb.iloc[maindish_indices].index
	
	#get user input recipe
	global usr_recipe 
	recipename = request.args.get('recipename')
	#print(recipename)
	usr_recipe = recipename.replace('_', ' ')
	#nutrfacts = recipedb.loc[usr_recipe]

	# get current BI of selected recipe
	current_bi = round(recipedb.loc[usr_recipe]['BI'], 2)
	global usr_recipe_bi
	usr_recipe_bi = f"LUNCH: '{usr_recipe}'    BI Score: {current_bi}"
	#usr_recipe_bi = "'" + str(usr_recipe) + "'" + '   BI Score: ' + str(current_bi)


	#save db indice for first choice for switching to recipe comparison pages
	np.save('app_data/firstchoice_ind',np.array([recipedb.index.get_loc(usr_recipe)]))

	#get class label for selected recipe
	optimal_recipe, optimal_bi = nd.optimal_recipe(usr_recipe)
	optimal_bi = round(optimal_bi, 2)
	global optimal_recipe_bi
	optimal_recipe_bi = f"DINNER: '{optimal_recipe}'    BI Score: {optimal_bi}"
	
	# plot bi score for lunch option and optimal dinner recipe with best BI score
	plot_url = nd.plot_bi_comparison(usr_recipe, optimal_recipe)		

	# get personalized recipe (locally optimal based on customer's customization)
	local_recipes = nd.local_recipe(usr_recipe, similarity)
	local_recipes.to_csv('app_data/local_recipes.csv', index=False)
	local_recipes['combine'] = "'" + local_recipes['recipename'].astype(str) + "'" + ' BI Score ' + local_recipes['BI_combine'].round(2).astype(str)	

	local_recipes_bi = local_recipes[['recipename', 'BI_combine']]	
	

	return render_template('firstchoice.html', 
				plot_url=plot_url,
	 		    	maindishes = maindishlist,
				okrecs = local_recipes['combine'],
				currentBI = usr_recipe_bi,
				bestBI = optimal_recipe_bi)


@app.route('/secondchoice', methods=['GET', 'POST'])
def secondchoice_comparefacts():
	'''Compare nutritional facts and ingredients of second and first recipe choice'''
	global similarity, allergy
	if request.method == 'POST':
	#if request.method == 'GET':
		for key, value in request.form.items():
			if key == 'similarity':
				similarity = float(value)
			if key == 'allergy' and value != '':
				allergy.append(value)
		print(similarity)
		print(allergy)
	#similarity = float(request.args.get('similarity'))	
	#indices for sample of 10 recipes from db from /index for sidebar
	maindish_indices = np.load('app_data/maindish_indices.npy')
	maindishlist = recipedb.iloc[maindish_indices].index
	
	# get personalized recipe (locally optimal based on customer's customization)
	global usr_recipe
	local_recipes = nd.local_recipe(usr_recipe, similarity)
	local_recipes.to_csv('app_data/local_recipes.csv', index=False)
	local_recipes['combine'] = "'" + local_recipes['recipename'].astype(str) + "'" + " BI Score " + local_recipes['BI_combine'].round(2).astype(str)	

	#name of selected second choice recipe
	recommended_recipe = request.args.get('recipename')
	print(recommended_recipe)
	if recommended_recipe is None:
		recommended_recipe = usr_recipe
		# recommended_recipe = optimal_recipe
	recommended_recipe = re.sub('_BI_Score_\d+.\d+', '', recommended_recipe).replace('_',' ')
	recommended_recipe = re.sub("'", '', recommended_recipe)
	print(recommended_recipe)

	plot_url = nd.plot_bi_comparison(usr_recipe, recommended_recipe)		

	# update the webpage
	global usr_recipe_bi, optimal_recipe_bi
	return render_template('secondchoice.html',
				plot_url = plot_url,
				maindishes = maindishlist,
				okrecs2 = local_recipes['combine'],
				currentBI = usr_recipe_bi,
				bestBI = optimal_recipe_bi,
				similarity = similarity)

#@app.route('/about')
#def about():
#	return render_template('about.html')

