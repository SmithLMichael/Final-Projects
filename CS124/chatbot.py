#!/usr/bin/env python
# -*- coding: utf-8 -*-

# PA6, CS124, Stanford, Winter 2018
# v.1.0.2
# Original Python code by Ignacio Cases (@cases) -- note from Michael: this relates to the 
# class structure primarily -- so TA's could relatively easily test everyone's projects automatically
# Final Project by Michael Smith and Cade May
######################################################################
import csv
import math
import string

import numpy as np
import re

from movielens import ratings
from random import randint
from PorterStemmer import PorterStemmer

class Chatbot:
    """Simple class to implement the chatbot for PA 6."""

    #############################################################################
    # `moviebot` is the default chatbot. Change it to your chatbot's name       #
    #############################################################################
    def __init__(self, is_turbo=False):
      self.name = 'moviebot69000'
      self.is_turbo = is_turbo

      self.stemmer = PorterStemmer()
      self.titles_lower_no_year = []
      self.title_map_lower = {}
      self.title_map = {}
      self.title_list = []
      self.titles_lower = []
      self.user_data = {}
      self.suggestions = {}
      self.datapoint_threshold = 4
      self.suggestion_mode = False
      self.shift = 1
      self.possible_titles_for_disambiguation = []
      self.disambiguating = False
      self.title_to_disambiguate = ""
      self.save_disambiguation_sentiment = 0.0 
      self.alt_title_map = {}
      self.read_data()
      # self.requesting_more_info_about_previous_movie = False

    #############################################################################
    # 1. WARM UP REPL
    #############################################################################

    def greeting(self):
      """chatbot greeting message"""
      #############################################################################
      # TODO: Write a short greeting message                                      #
      #############################################################################

      greeting_message = "Hey Friend!\nTell me about some movies you've seen and I'll suggest some new ones you might like."

      #############################################################################
      #                             END OF YOUR CODE                              #
      #############################################################################

      return greeting_message

    def goodbye(self):
      """chatbot goodbye message"""
      #############################################################################
      # TODO: Write a short farewell message                                      #
      #############################################################################

      goodbye_message = 'Have a nice day!'

      #############################################################################
      #                             END OF YOUR CODE                              #
      #############################################################################

      return goodbye_message


    #############################################################################
    # 2. Modules 2 and 3: extraction and transformation                         #
    #############################################################################
    def process(self, input):
      """Takes the input string from the REPL and call delegated functions
      that
        1) extract the relevant information and
        2) transform the information into a response to the user
      """
      #############################################################################
      # TODO: Implement the extraction and transformation in this method, possibly#
      # calling other functions. Although modular code is not graded, it is       #
      # highly recommended                                                        #
      #############################################################################
  
      if self.is_turbo == True:
	''' creative '''

	if self.suggestion_mode:
		affirmativeAnswers = ["yes", "Yes", "sure", "y", "yes please", "okay"]
		if input in affirmativeAnswers:
			response = self.suggest()
		else:
			responses = ["Sorry what?", "Huh?", "I know enough about you now.", "That was a yes or :quit question."]
			ask = " Would you like to hear another recommendation? (Or enter :quit if you're done)"
			response = responses[randint(0, len(responses) - 1)] + ask

	else:


		if not self.disambiguating:

			movie_title, sentiment = self.get_relevant_info_turbo(input)
			
			# check for error strings and then return proper response
			
			if isinstance(movie_title, str):
				if movie_title[0:6] == "ERROR_":
					''' determine error type and respond accordingly '''
					response = self.get_error_response(movie_title, input)	

				elif movie_title == "COMMENCE_DISAMBIGUATION":
					
					response = "Which " + self.title_to_disambiguate + " movie do you mean? "
					self.save_disambiguation_sentiment = sentiment
					self.disambiguating = True
				else:
					''' valid movie title, so construct a response about it '''
			

					# get movie response, and save user data if sentiment is meaningful (nonzero)
					response = self.get_movie_response_turbo(movie_title, sentiment, input)
					
					# if we have enough data on the user, start suggesting movies
					if len(self.user_data) >= self.datapoint_threshold:
						self.suggestion_mode = True
						response += self.suggest()

			else:
				# multiple titles
				title1 = movie_title[0]
				title2 = movie_title[1]
		
				sentiment1 = sentiment[0]
				sentiment2 = sentiment[1]
	
				
				response = self.get_multiple_movie_response(title1, title2, sentiment1, sentiment2)


				if len(self.user_data) >= self.datapoint_threshold:
					self.suggestion_mode = True
					response += self.suggest()
					
	
		else:
			# disambiguating	
			dis_result = self.disambiguate_part_2(input)
			
			if dis_result == "ERROR_UNKNOWN_TITLE":
				response = self.get_error_response("ERROR_UNKNOWN_TITLE", input)
			else:
				response = self.get_movie_response_turbo(dis_result, self.save_disambiguation_sentiment, input)

		        self.possible_titles_for_disambiguation = []
		        self.disambiguating = False
		        self.title_to_disambiguate = ""
		        self.save_disambiguation_sentiment = 0.0
	
      else:
	''' baseline  '''
	if self.suggestion_mode:
		affirmativeAnswers = ["yes", "Yes", "sure"]
		if input in affirmativeAnswers:
			response = self.suggest()
		else:
			response = "Sorry what?"	

	else:
		movie_title, sentiment = self.get_relevant_info(input)

		# check for error strings and then return proper response
		
		
		if movie_title[0:6] == "ERROR_":
			''' determine error type and respond accordingly '''
			response = self.get_error_response(movie_title)	
		else:
			''' valid movie title, so construct a response about it '''
		
			# get movie response, and save user data if sentiment is meaningful (nonzero)
			response = self.get_movie_response(movie_title, sentiment)
			
			# if we have enough data on the user, start suggesting movies
			if len(self.user_data) >= self.datapoint_threshold:
				self.suggestion_mode = True
				response += self.suggest()	
			

      return response
  


	 
    def suggest(self):
	
	r = ""
	if len(self.suggestions) == 0:
		suggestions = self.get_suggestions()
		self.suggestions = suggestions
	else:
		suggestions = self.suggestions



	suggestionIndex = suggestions[len(suggestions) - self.shift][1]
	suggestion = self.titles[suggestionIndex][0]

	r += "\nThat's enough for me to make a recommendation.\n"
	r += "I suggest you watch " + suggestion + ".\n"
	r += "Would you like to hear another recommendation? (Or enter :quit if you're done.)"
	self.shift += 1	
	return r

    def get_multiple_movie_response(self, title1, title2, sentiment1, sentiment2):

	if sentiment1 > 0:
		self.user_data[title1] = 1.0
		response = self.get_pos_response(title1)
		
	elif sentiment1 < 0:

		self.user_data[title1] = -1.0
	
		response = self.get_neg_response(title1)
	else:	
		response = "I'm not quite sure how you felt about " + title1 + "."
		response += "\nPlease tell me more about " + title1 + "."

	response += "   "
	if sentiment2 > 0:
		self.user_data[title2] = 1.0
		response += self.get_pos_response(title2)
		
	elif sentiment2 < 0:

		self.user_data[title2] = -1.0
	
		response += self.get_neg_response(title2)
	else:


			
		response += "I'm not quite sure how you felt about " + title2 + "."
		response += "\nPlease tell me more about " + title2 + "."


	return response



    def disambiguate_part_2(self, addition):


	for title in self.possible_titles_for_disambiguation:
		
		if addition in title:
	
			return title

	return "ERROR_UNKNOWN_TITLE"
		
 
    def get_suggestions(self):
      return self.recommend(self.user_data)

    def get_relevant_info(self, s):
	
      title, sentence_without_title = self.get_title(s)	

      if title == "ERROR_NO_TITLE":
	return ("ERROR_NO_TITLE", "ERROR_NO_TITLE")
	
      if title == "ERROR_MULTIPLE_TITLES":
	return ("ERROR_MULTIPLE_TITLES", "ERROR_MULTIPLE_TITLES")

      if title == "ERROR_UNKNOWN_TITLE":
	return ("ERROR_UNKNOWN_TITLE", "ERROR_UNKNOWN_TITLE") 
	
      sentiment = self.extract_sentiment(sentence_without_title)

      return (title, sentiment)

    def get_relevant_info_turbo(self, s):
      title, sentence_without_title = self.get_title_turbo(s)	

	
      if isinstance(title, str):

	      if title == "ERROR_NO_TITLE":
		return ("ERROR_NO_TITLE", "ERROR_NO_TITLE")
		
	      if title == "ERROR_MULTIPLE_TITLES":
		return ("ERROR_MULTIPLE_TITLES", "ERROR_MULTIPLE_TITLES")

	      if title == "ERROR_UNKNOWN_TITLE":
		return ("ERROR_UNKNOWN_TITLE", "ERROR_UNKNOWN_TITLE") 
		

	      sentiment = self.extract_sentiment_turbo(sentence_without_title)
		
	      if title == "COMMENCE_DISAMBIGUATION":
		return ("COMMENCE_DISAMBIGUATION", sentiment)

      else:
	      sentiment = self.extract_multiple_sentiment(sentence_without_title)
	      return (title, sentiment)	
	
      return (title, sentiment)

    def extract_multiple_sentiment(self, s):
	score1 = 0.0
	score2 = 0.0



	separate = re.findall("MOVIE1,[^.]*?(but|however|yet|though)[^.]*?MOVIE2", s)
	
	if len(separate) > 0:	
		independent_sentiments = True
	else:
		independent_sentiments = False

	if independent_sentiments == False:
		
		score1 = self.sentiment_score_derive(s)
		score2 = score1
	
	else:
		# separate the sentences:
		
		for i in range(len(s)):
			if s[i] == ",":
				if i > s.find("MOVIE1") and i < s.find("MOVIE2"):
					divider_comma_index = i
			
		movie1_sentence = s[0:divider_comma_index]
		movie2_sentence = s[divider_comma_index:]
		
		score1 = self.sentiment_score_derive(movie1_sentence)
		score2 = self.sentiment_score_derive(movie2_sentence)
	
				
		
				

	return((score1, score2))
    
    def sentiment_score_derive(self, s):
	score1 = 0.0
	s = s.translate(None, string.punctuation)	
	words = s.split(' ')

	disqualifiers = ["not", "didn't", "never", "didnt", "don't", "dont"]
	sentiments = []
	
	# get sentiment values
	for w in words:
		sentiment = "---"
			
		stem = self.stemmer.stem(w)
		if stem in self.sentiment:
			sentiment = self.sentiment[stem]

		sentiments.append(sentiment)

	# negation
	for i in range(len(words) - 1):
		w = words[i]
		if w in disqualifiers:

			if sentiments[i + 1] == "pos":
				sentiments[i + 1] = "neg"
			elif sentiments[i + 1] == "neg":
				sentiments[i + 1] = "pos"

			if (i + 2) < len(words):
				if sentiments[i + 2] == "pos":
					sentiments[i + 2] = "neg"
				elif sentiments[i + 2] == "neg":
					sentiments[i + 2] = "pos"

	for s in sentiments:			
		if s == "pos":
			score1 += 1
		if s == "neg":
			score1 += -1

	if score1 > 0:
		score1 = 1.0
	if score1 < 0:
		score1 = -1.0
	return score1

    def extract_sentiment_turbo(self, s):

	score = 0.0
	
	any_pos = False
	any_neg = False

	strong_pos = False
	strong_neg = False
	

	strong_pos_list = ["love", "favorite", "amazing", "fantastic", "wonderful", "incredible"]
	strong_neg_list = ["hate", "horrible", "terrible"]
		
	strong_pos_list = [self.stemmer.stem(x) for x in strong_pos_list]
	strong_neg_list = [self.stemmer.stem(x) for x in strong_neg_list]



	enhancers = ["v+e+r+y+", "r+e+a+l+l+y+", "e+x+t+r+e+m+e+l+y+", "t+o+t+a+l+l+y+"]
	enhance_regex = '(' + '|'.join(enhancers) + ')'
	
	disqualifiers = ["not", "didn't", "never", "didnt", "don't", "dont"]
	s = s.translate(None, string.punctuation)	
	words = s.split(' ')
	
	sentiments = []
	
	# get sentiment values
	for w in words:
		sentiment = "---"
			
		stem = self.stemmer.stem(w)
		if stem in self.sentiment:
			sentiment = self.sentiment[stem]

		sentiments.append(sentiment)

	# negation
	for i in range(len(words) - 1):
		w = words[i]
		if w in disqualifiers:

			if sentiments[i + 1] == "pos":
				sentiments[i + 1] = "neg"
			elif sentiments[i + 1] == "neg":
				sentiments[i + 1] = "pos"

			if (i + 2) < len(words):
				if sentiments[i + 2] == "pos":
					sentiments[i + 2] = "neg"
				elif sentiments[i + 2] == "neg":
					sentiments[i + 2] = "pos"

	# check for extremes

	for word in words:
		if self.stemmer.stem(word) in strong_pos_list:
			strong_pos = True
		if self.stemmer.stem(word) in strong_neg_list:
			strong_neg = True

	
	# check for enhancements		
	for i in range(len(words) - 1):
		w = words[i]
		
		match = re.findall(enhance_regex, w)
		
		if len(match) > 0:

			if sentiments[i + 1] == "pos":
				strong_pos = True
			if sentiments[i + 1] == "neg":
				strong_neg = True


		
		
	# calculate score
	for s in sentiments:			
		if s == "pos":
			any_pos = True
			score += 1
		if s == "neg":
			any_neg = True
			score += -1
	
	
	if score > 0:
		if strong_pos and not any_neg:
			#really like
			return 10
		if strong_pos and any_neg:
			# really like, but mixed feelings
			return 5
		
		score = 1.0
	if score < 0:

		if strong_neg and not any_pos:
			# really dislike
			return -10

		if strong_neg and any_pos:
			# really dislike, but mixed feelings
			return -5

		score = -1.0
	
	return score



    def extract_sentiment(self, s):

	score = 0.0

	disqualifiers = ["not", "didn't", "never", "didnt", "don't", "dont"]
	s = s.translate(None, string.punctuation)	
	words = s.split(' ')
	
	sentiments = []
	
	# get sentiment values
	for w in words:
		sentiment = "---"
			
		stem = self.stemmer.stem(w)
		if stem in self.sentiment:
			sentiment = self.sentiment[stem]

		sentiments.append(sentiment)

	# negation
	for i in range(len(words) - 1):
		w = words[i]
		if w in disqualifiers:

			if sentiments[i + 1] == "pos":
				sentiments[i + 1] = "neg"
			elif sentiments[i + 1] == "neg":
				sentiments[i + 1] = "pos"

			if (i + 2) < len(words):
				if sentiments[i + 2] == "pos":
					sentiments[i + 2] = "neg"
				elif sentiments[i + 2] == "neg":
					sentiments[i + 2] = "pos"

	for s in sentiments:			
		if s == "pos":
			score += 1
		if s == "neg":
			score += -1

	if score > 0:
		score = 1.0
	if score < 0:
		score = -1.0
	
	return score


    def get_title(self, s):

	numQuoteMarks = s.count("\"")
	
	if numQuoteMarks < 2:
		return ("ERROR_NO_TITLE", s)

	if numQuoteMarks > 2:
		return ("ERROR_MULTIPLE_TITLES", s)
	

	first_quote_index = s.find("\"")	
	second_quote_index = s.find("\"", first_quote_index + 1)
	
	# without quotes
	title = s[first_quote_index + 1 : second_quote_index]
	
	# check to see if title is in database
        if title not in self.title_map.keys():
		articled_title = self.article_shift_title(title)
	
		if articled_title in self.title_map.keys():
			title = articled_title
		else:
			return ("ERROR_UNKNOWN_TITLE", s)

	# replace movie name with "MOVIE"
	titleWithQuotes = s[first_quote_index : second_quote_index + 1]
	clean_sentence = s.replace(titleWithQuotes, "MOVIE")

	return (title, clean_sentence)

    def get_title_turbo(self, s):

	numQuoteMarks = s.count("\"")

	if numQuoteMarks < 2:

		return self.identify_title_turbo(s)
		#return ("ERROR_NO_TITLE", s)

	if numQuoteMarks > 2:
		
		if numQuoteMarks == 4:
			return self.multiple_titles(s)
		return ("ERROR_MULTIPLE_TITLES", s)


	first_quote_index = s.find("\"")	
	second_quote_index = s.find("\"", first_quote_index + 1)

	# without quotes
	title = s[first_quote_index + 1 : second_quote_index]

	# check to see if title is in database
	if title not in self.title_map.keys():
		articled_title = self.article_shift_title(title)

		if articled_title in self.title_map.keys():
			title = articled_title
		elif title in self.alt_title_map:
			title = self.alt_title_map[title]
		else:

			disambiguation = self.disambiguate(title, s)
			
			if disambiguation[1] == "ERROR_NO_TITLE_STUB":
				spelling_correction = self.spell_correct(title, s)
				
				if spelling_correction[1] == "ERROR_UNKNOWN_TITLE":
					return self.identify_title_turbo(s)
				else:
					return spelling_correction

			else:
				return self.disambiguate(title, s)

	# replace movie name with "MOVIE"
	titleWithQuotes = s[first_quote_index : second_quote_index + 1]
	clean_sentence = s.replace(titleWithQuotes, "MOVIE")
	
	return (title, clean_sentence)


    def multiple_titles(self, s):

	first_quote_index = s.find("\"")	
	second_quote_index = s.find("\"", first_quote_index + 1)

	third_quote_index = s.find("\"", second_quote_index + 1)
	fourth_quote_index = s.find("\"", third_quote_index + 1)

	title1 = s[first_quote_index + 1 : second_quote_index]
	title2 = s[third_quote_index + 1 : fourth_quote_index]

	# check to see if titles in database

        if title1 not in self.title_map.keys():
		articled_title1 = self.article_shift_title(title1)
	
		if articled_title1 in self.title_map.keys():
			title1 = articled_title1
		else:
			title1, j = self.identify_title_turbo(title1)
			if title1 == "ERROR_NO_TITLE": 
				return ("ERROR_UNKNOWN_TITLE", s)

	if title2 not in self.title_map.keys():
		articled_title2 = self.article_shift_title(title2)
	
		if articled_title2 in self.title_map.keys():
			title2 = articled_title2
		else:
			title2, j = self.identify_title_turbo(title2)
			if title2 == "ERROR_NO_TITLE": 
				return ("ERROR_UNKNOWN_TITLE", s)


	title1WithQuotes = s[first_quote_index : second_quote_index + 1]
	title2WithQuotes = s[third_quote_index : fourth_quote_index + 1]

	clean_sentence = s.replace(title1WithQuotes, "MOVIE1")	
	clean_sentence = clean_sentence.replace(title2WithQuotes, "MOVIE2")
	
	return ((title1, title2), clean_sentence)




    def spell_correct(self, title, s):
	title = title.lower()
	titleWords = title.split()

	
	working_dict = {tit[0]:0 for tit in self.titles_lower_no_year if len(tit[0].split()) == len(titleWords)}

	d = {tit[0]:0 for tit in self.titles_lower if len(tit[0].split()) == len(titleWords)}
	
	working_dict.update(d)
	

	for i in range(len(titleWords)):
		curr_dict = {}
		for key in working_dict:
			# compute distance
			editDist = self.compute_edit_distance(key.split()[i], titleWords[i])
			if editDist < 3: # vary this? - this currently says that 
				curr_dict[key] = working_dict[key] + editDist
		working_dict = curr_dict

	# This will give the title with the minimum edit distance - but we can also think about
	# working with a few to disambiguate?
	if len(working_dict) == 0:
		return ("ERROR_UNKNOWN_TITLE", "ERROR_UNKNOWN_TITLE")

	title = min(working_dict, key=working_dict.get)

	clean_sentence = re.sub("\"" + title + "\"", "", s)


	if title in self.title_map_lower.keys():
		genres = self.title_map_lower[title]
		tup = [title, genres]
	else:
		genres = self.title_map_lower[self.article_shift_title_wout_year(title)]
		tup = [self.article_shift_title_wout_year(title), genres]

	if tup in self.titles_lower_no_year:
		index = self.titles_lower_no_year.index(tup)
	else:
		index = self.titles_lower.index(tup)	

	official_title = self.titles[index][0]
	
	return (official_title, clean_sentence)


    def compute_edit_distance(self, keyword, word):
    
    	m = len(keyword) + 1
        n = len(word) + 1
        d = np.zeros((m, n))
        
        for i in range(m):
            d[i][0] = i
    
        for j in range(n):
            d[0][j] = j
        
        for j in range(1, n):
            for i in range(1, m):
                if keyword[i-1] == word[j-1]:
                    d[i, j] = d[i-1, j-1]
                else:
                	# insertion, deletion, substitution
                	d[i, j] = min(d[i-1, j] + 1, d[i, j-1] + 1, d[i-1, j-1] + 1)
        
        
	return d[m-1,n-1]






    	
    def disambiguate(self, title, s):
	
	for potential_title in self.title_map.keys():
		if title in potential_title:
			self.possible_titles_for_disambiguation.append(potential_title)
		
	
	if len(self.possible_titles_for_disambiguation) < 2:
		return ("ERROR_NO_TITLE_STUB", "ERROR_NO_TITLE_STUB")

	# clean title out of sentence
	clean_sentence = re.sub("\"" + title + "\"", "", s)
	self.title_to_disambiguate = title 

	return ("COMMENCE_DISAMBIGUATION", clean_sentence)
		
	
	
    def identify_title_turbo(self, s):
	possible_titles = []
	l = s.lower()

	for t in self.title_list:
		lt = t.lower()
		if lt in l:
			possible_titles.append(lt)		

	if len(possible_titles) == 0:
		return ("ERROR_NO_TITLE", s)

	
	# filter out internal "titles", like finding "o" in "harry potter"
	placeholder = []
	for po in possible_titles:
		#filter_regex = "(\w" + po + "\w)"
		filter_regex = "([\w]" + po + "|" + po + "[\w])"

		matches = re.findall(filter_regex, s)
		
		if len(matches) == 0:
			placeholder.append(po)

	possible_titles = placeholder
	
	
	if len(possible_titles) == 0:
		return ("error_no_title", s)
	

	
	# sort titles
	tit = [[len(x), x] for x in possible_titles] 
	tit = sorted(tit)
	title = tit[len(tit) - 1][1]
	
	if title in self.title_map_lower.keys():
		genres = self.title_map_lower[title]
		tup = [title, genres]
	else:
		genres = self.title_map_lower[self.article_shift_title_wout_year(title)]
		tup = [self.article_shift_title_wout_year(title), genres]
	

	
	year = re.findall("(\(\d\d\d\d\))$", tup[0])
	if len(year) > 0:
		tup = [tup[0][0:len(tup[0]) - 7], tup[1]]	
	


 
	index = self.titles_lower_no_year.index(tup)
	official_title = self.titles[index][0]

	

	# clean sentence 
	clean_sentence = re.sub(title, "", l)
		
	return (official_title, clean_sentence)


    def article_shift_title(self, title):

	articles = ["the", "an", "a"]

	if title.count(' ') < 1:
  		return title
  
	words = title.split(' ')
	article = words[0]
	
	if article.lower() not in articles:
		return title

	articled_title = ""

	for i in range(1, len(words) - 2):
  		articled_title += words[i] + " "

	articled_title += words[len(words) - 2]
  
	articled_title += ", " + article
	articled_title += " " + words[len(words) - 1]

	return articled_title


    def article_shift_title_wout_year(self, title):
	articles = ["the", "an", "a"]

	if title.count(' ') < 1:
  		return title
  
	words = title.split(' ')
	article = words[0]
	
	if article.lower() not in articles:
		return title

	articled_title = ""

	for i in range(1, len(words) - 1):
  		articled_title += words[i] + " "

	articled_title += words[len(words) - 1]
  
	articled_title += ", " + article

	
	return articled_title


    def get_movie_response_turbo(self, movie_title, sentiment, s):

	if sentiment > 0:
		self.user_data[movie_title] = 1.0

		if sentiment == 10:
			# really like
			return self.super_pos_response(movie_title)
		
		if sentiment == 5:
			# really like, mixed feelings
			return self.super_pos_response_mixed(movie_title)

		response = self.get_pos_response(movie_title)
		
	elif sentiment < 0:

		self.user_data[movie_title] = -1.0
		
		if sentiment == -10:
			# really dislike
			return self.super_neg_response(movie_title)
		if sentiment == -5:
			# really dislike, mixed feelings
			return self.super_neg_response(movie_title)

		response = self.get_neg_response(movie_title)
	else:
			
		s_is_question_about_primary_title = self.determine_if_asking_about_alternate_title(s)

		if s_is_question_about_primary_title:

			year = re.findall(r'\([0-9]{4}\)', movie_title)
		        movie = re.sub(' \(.*?\)', '', movie_title)
		        article = re.findall('(?:, The|, An|, A)', movie)
		  	if len(article) != 0:
				article = article[0][2:]
				movie = article + ' ' + movie[0:len(movie)-len(article)-2]

			response = "The primary title is " + movie + ' ' + year[0] +  "."

	
		else:	
			response = "Hmm. I'm not quite sure how you felt about " + movie_title + "."
			response += "\nPlease tell me more about " + movie_title + "."

	return response


    def determine_if_asking_about_alternate_title(self, s):
	search_regex = "(?:What else is \".*?\" (?:called|named)|Give me another name for \".*?\"|(?:What's|What is) (?:another name|the primary title) for \".*?\")"
	if len(re.findall(search_regex, s, flags=re.IGNORECASE)) != 0:
		return True
	else:
		return False
	

    def ret_primary_title(self, alt_title_name):	

	    full_movie = self.alt_title_map[alt_title_name]
	    year = re.findall(r'\([0-9]{4}\)', full_movie)
	    movie = re.sub(' \(.*?\)', '', full_movie)

	    article = re.findall('(?:, The|, An|, A)', movie)
	    if len(article) != 0:
		article = article[0][2:]
		movie = article + ' ' + movie[0:len(movie)-len(article)-2]

	    return movie + ' ' + year[0]


    def super_pos_response(self, title):
 
	responses = ["Wow you really really liked %s! Awesome!"]
	ind = randint(0, len(responses) - 1)
		
	return responses[ind] % title

    def super_pos_response_mixed(self, title):
	
	responses = ["Sounds like you really really enjoyed %s, but you had a bit of a mixed experience. Cool!"]

	ind = randint(0, len(responses) - 1)
		
	return responses[ind] % title
   
    def super_neg_response(self, title):

	responses = ["Wow! You did NOT like %s! Sounds like you really hated that one."]
	
	ind = randint(0, len(responses) - 1)
		
	return responses[ind] % title

    
    def super_neg_response(self, title):

	responses = ["Jeez. Sounds like you really really didn't like %s, but there were maybe a few good things about it."]
	
	ind = randint(0, len(responses) - 1)
		
	return responses[ind] % title



    def get_movie_response(self, movie_title, sentiment):

	if sentiment > 0:
		self.user_data[movie_title] = sentiment
		response = self.get_pos_response(movie_title)
		
	elif sentiment < 0:
		self.user_data[movie_title] = sentiment
		response = self.get_neg_response(movie_title)
	else:	
		response = "Hmm. I'm not quite sure how you felt about " + movie_title + "."
		response += "\nPlease tell me more about " + movie_title + "."
		# self.requesting_more_info_about_previous_movie = True

	return response



    def get_error_response(self, error, user_input):
	
	response = ""

	if error == "ERROR_NO_TITLE":
		
		user_is_talking_about_movie = self.determine_user_intentions(user_input)

		if user_is_talking_about_movie:	
			response = self.get_no_title_response()
		else:
			response = self.conversate(user_input) 
			
	if error == "ERROR_MULTIPLE_TITLES":
		response = self.get_multiple_title_response()
			
	if error == "ERROR_UNKNOWN_TITLE":
		response = self.get_unknown_title_response()


	return response

    def conversate(self, user_input):
	
	responses = ["Haha. Interesting.", "Neat!", "Nice.", "Cool story bro.", "That's fascinating.", "Cool beans!"]
	
	ind = randint(0, len(responses) - 1)
	response = responses[ind]
	
	
	hi = "([hH]i|[hH]ey|[hH]owdy|[hH]ola|[hH]ello)"
	matches = re.findall(hi, user_input)
	
	if len(matches) > 0:
		responses = ["Hi there.", "Hey.", "Howdy.", "Sup.", "Greetings."]
		ind = randint(0, len(responses) - 1)
		response = responses[ind]
		return response


		
	
	# can questions
	r = "^[Cc]an ([^.\n?]*)"
	matches = re.findall(r, user_input)
	
	if len(matches) > 0:
		question_content = matches[0]
		question_content = question_content.split()
		
		for i in range(len(question_content)):
			word = question_content[i]
			
			if re.search("[yY]ou", word) != None:
				question_content[i] = "I"
					
			if re.search("[Ii]", word) != None:
				question_content[i] = "you"

						
			if re.search("[tT]his", word) != None:
				question_content[i] = "that"
			
		
			if re.search("[mM]e", word) != None:
				question_content[i] = "you"

		
		response = "I don't know. Can " + ' '.join(question_content) + "?"

	# what questions
	r = "^[wW]hat [\w]* ([^.\n?]*)"
	matches = re.findall(r, user_input)

	if len(matches) > 0:

		responses = ["Great question. Sorry I don't know the answer.",
				 "Ask again later.", "Concentrate and ask again.",
					 "I'd better not tell you now.",
					 "Reply hazy try again", "What is this? 20 questions?",
					"Sorry, I'm not sure about that.", "Ask again later."]

		ind = randint(0, len(responses) - 1)
		response = responses[ind]
	
		"""
		question_content = matches[0]
		question_content = question_content[:1].upper() + question_content[1:]
		response = question_content + " is something I don't know very much about."	
		"""

	return response


    def determine_user_intentions(self, user_input):

	numQuoteMarks = user_input.count("\"")
	
	if numQuoteMarks == 2:
		return True


	return False


 
    def get_pos_response(self, title):
	
	responses = ["You liked %s. Thank you!",
			 "Nice! I'm glad you enjoyed %s!",
			 "Cool beans! Glad ya liked %s ;)",
			 "Sweet! I liked %s too! :p",
			 "Neato! Agreed, %s was an awesome movie!"]
	
	ind = randint(0, len(responses) - 1)
		
	return responses[ind] % title

    def get_neg_response(self, title):
	
	responses = ["You didn't like %s. Thank you!", 
			"Aw bummer. I'm sorry you didn't like %s.",
			 "Phooey! %s sounds like a real bummer.", 
			"Aw schucks! I liked %s. I'm sorry you didn't!"]
	
	
	ind = randint(0, len(responses) - 1)
		
	return responses[ind] % title


    def get_no_title_response(self):
	responses = ["Sorry, I don't understand. Tell me about a movie that you have seen.",
			 "It'd be great if you told me about a movie.", 
			"You good bro? Tell me about a movie?",
			 "I'm here to talk about movies, not your problems. Tell me about a movie. :)", 
			"Okay... What's a movie you like/dislike? How did it make you feel?"]

 
	ind = randint(0, len(responses) - 1)
	
	return responses[ind]

    def get_multiple_title_response(self):
	responses = ["Please tell me about one movie at a time. Go ahead.", 
			"Woah woah woah. Chill. One at a time!", 
			"Easy there cowboy! I like to talk about one movie at a time.",
			 "I prefer talking about one movie at a time!"]


	ind = randint(0, len(responses) - 1)
	
	return responses[ind]

    def get_unknown_title_response(self):
	responses = ["Sorry, I'm not understanding. Tell me about a movie that you have seen.", 
		"I don't think I've seen that one yet. Tell me about another movie.", 
		"Hmm. I haven't heard of that movie. Talk to me about another movie.", 
		"What an interesting title. I'm not familiar with it, unfortunately. Give me another movie."]
	ind = randint(0, len(responses) - 1)
	
	return responses[ind]
	
		
    #############################################################################
    # 3. Movie Recommendation helper functions                                  #
    #############################################################################

    def read_data(self):
      """Reads the ratings matrix from file"""
      # This matrix has the following shape: num_movies x num_users
      # The values stored in each row i and column j is the rating for
      # movie i by user j
      self.titles, self.ratings = ratings()
      reader = csv.reader(open('data/sentiment.txt', 'rb'))
      self.sentiment = dict(reader)
      
      # binarize ratings
      self.binarize()
      self.create_title_map()
      self.stem_sentiment()

    def binarize(self):
      """Modifies the ratings matrix to make all of the ratings binary"""

      threshold = 2.5
      
      for i in range(len(self.ratings)):
	for j in range(len(self.ratings[0])):
		
		rating = float(self.ratings[i][j])
		
		if rating == 0:
			self.ratings[i][j] = 0	
		elif rating < threshold:
			self.ratings[i][j] = -1.0
		else: 
			self.ratings[i][j] = 1.0

	
    def article_deshift_title(self, movie):

	w = movie.split(' ')

	# remove comma
	w[len(w) - 2] = w[len(w) - 2][0:len(w[len(w) - 2]) - 1]
	# shift article to front
	w.insert(0, w[len(w) - 1])

	w = w[0:len(w) - 1]

	title = ' '.join(w)
	
	return title

    def alternate_titles(self, movie):
	alt_titles = re.findall('\((?:a\.k\.a )?([A-Za-z].*?)\)', movie)
    	# takes care of the a.k.a.
   	for i in range(len(alt_titles)):
       		alt_titles[i] = re.sub('a.k.a. ', '', alt_titles[i])
            
       		 # article @ back, no year
        	self.alt_title_map[alt_titles[i]] = movie
                
                # article @ back, year
        	self.alt_title_map[alt_titles[i] + movie[len(movie) - 7:]] = movie
                    
                article = re.findall(', (.*)', alt_titles[i])
                if len(article) != 0:
                	article = article[len(article)-1]
                	if len(article) <= 3:
                		article_front = alt_titles[i]
                		article_front = article_front[0:len(article_front)-len(article)-2]
                                        
                                if article == "L'" or article == "l'":
                                	article_front = article + article_front
                                else:
                                	article_front = article + ' ' + article_front
                                        
                                self.alt_title_map[article_front] = movie
                                self.alt_title_map[article_front + movie[len(movie) - 7:]] = movie
	
    def create_title_map(self):
	
	permutations= []
	for t in self.titles:
		movie = t[0]
		genres = t[1]
		
		self.alternate_titles(movie)	
	
		without_year_movie = movie[0:len(movie) - 7]
		permutations.append(without_year_movie)

		articleCheck = re.findall("(, ?(?:[Tt]he|[Aa]n|[Aa]))$", without_year_movie)

		
		articled_title = ""
		if len(articleCheck) > 0:
			articled_title_no_year = self.article_deshift_title(without_year_movie)
			permutations.append(articled_title_no_year)
			
		"""	
		if articled_title != without_year_movie and articled_title != "":
			#without_year_articled = articled_title[0:len(articled_title) - 7]
			permutations.append(articled_title)
			permutations.append(without_year_articled)
		"""
		
		self.title_map[movie] = genres

	self.title_list = self.title_map.keys()
	self.title_list.extend(permutations)
	self.title_list = [xx.lower() for xx in self.title_list]

	self.titles_lower = [[xx[0].lower(), xx[1]] for xx in self.titles]
	self.titles_lower_no_year = [[x[0][0:len(x[0])-7], x[1]]for x in self.titles_lower]
	
	
	"""
	x = sorted(self.titles_lower_no_year)
	for a in x:
		print a

	"""
	"""
	l = [[len(xx), xx]for xx in self.title_list]
	l = sorted(l)
	for a in l:
		print a
	"""

	for t in self.titles_lower:
		movie = t[0]
		genres = t[1]
		without_year_movie = movie[0:len(movie) - 7]
		
		self.title_map_lower[without_year_movie] = genres
		self.title_map_lower[movie] = genres

	
    def stem_sentiment(self):
	
	replacement = {}
	for word, sent in self.sentiment.iteritems():
		replacement[self.stemmer.stem(word)] = sent

	self.sentiment = replacement
		


    def distance(self, u, v):
      """Calculates a given distance function between vectors u and v"""
      # TODO: Implement the distance function between vectors u and v]
      # Note: you can also think of this as computing a similarity measure

      if np.linalg.norm(u) != 0 and np.linalg.norm(v) != 0:
	return float(np.dot(u,v)) / (np.linalg.norm(u) * np.linalg.norm(v))
      else:
	return 0.0

    def recommend(self, u):
      """Generates a list of movies based on the input vector u using
      collaborative filtering"""
      # TODO: Implement a recommendation function that takes a user vector u
      # and outputs a list of movies recommended by the chatbot

      ratings = []
      indeces = set()
      

      # for movie in movies
      for i in range(len(self.titles)):

	total = 0.0
      	for user_rated_movie, rating in u.iteritems():

		r1 = self.ratings[i]

		genres = self.title_map[user_rated_movie]
		tup = [user_rated_movie, genres]
		index = self.titles.index(tup)
		indeces.add(index)

		r2 = self.ratings[index]

		s = self.distance(r1, r2)

		total += s * rating	

	if i not in indeces:
		ratings.append([total, i])
	else:
		ratings.append([-100.1, i])


      return sorted(ratings)
    
    #############################################################################
    # 4. Debug info                                                             #
    #############################################################################

    def debug(self, input):
      """Returns debug information as a string for the input string from the REPL"""
      # Pass the debug information that you may think is important for your
      # evaluators
      debug_info = 'debug info'
      return debug_info


    #############################################################################
    # 5. Write a description for your chatbot here!                             #
    #############################################################################
    def intro(self):
      return """ 
      Welcome to moviebot.\n
      Our luxurious, creative features include:\n
      \n
      * Identifying movies without quotation marks or perfect capitalization\n
      * Fine-grained sentiment extraction\n
      * Spell-checking movie titles\n
      * Disambiguating movie titles for series and year ambiguities\n
      * Responding to arbitrary input\n
      * Alternate/foreign titles\n
      * Extracting sentiment with multiple-movie input\n
      * You can ask: What else is "alt title" (called|named)? Give me another name for "alt title". What's the primary name for "alt title"?\n
      * ^ This allows you to input an alternate/foreign title, and get the 'primary' title back.
      """



    #############################################################################
    # Auxiliary methods for the chatbot.                                        #
    #                                                                           #
    # DO NOT CHANGE THE CODE BELOW!                                             #
    #                                                                           #
    #############################################################################

    def bot_name(self):
      return self.name


if __name__ == '__main__':
    Chatbot()
