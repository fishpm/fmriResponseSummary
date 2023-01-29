#!/usr/bin/python3

### Script processes fMRI behavioral response files and produces event-by-event and summary files. Flexible to process many different types of behavioral tasks. Returns .csv event-by-event and summary and .mat files that can be used in SPM for single-subject analyses.
### Input can be a 1) single file, 2) MRI scanner name, 3) MRI scanner ID (incomplete IDs partially supported), 4) .txt file with individual elements on each line. Elements in 4) can be of any type 1)-3).

### Example commands
## python3 fmriResponseSummary_RH.py /rawdata/mr-rh/MRraw/prisma/p086_ln/faces2_p086_ln_1763.txt
## python3 fmriResponseSummary_RH.py faces2_p086_ln_1763.txt (expects file in current dir)
## python3 fmriResponseSummary_RH.py p123
## python3 fmriResponseSummary_RH.py p1 (will process all files for all subjects starting with "p1", e.g., p12, p100, p199, etc.)
## python3 fmriResponseSummary_RH.py prisma (will process all files for all prisma scan folders)

### Tasks supported
## Hariri Faces Task (20180109); updated 20220707
## Reward Task (20180109); updated 20220706
## Alcohol Task (20180109)
## PSAP (20220708)
## RLP (20220628)
## Breathing (20230117)
## Loss Aversion (20230122) - CBS Entrepreneurship

### Updates
## 20201111 - work with MR001 datas

### Patrick M. Fisher

# Load relevant libraries
from sys import argv
#from sys import exit
import re
import pandas as pd
import numpy as np
from os import listdir
from os import path
from io import open
import os
import io
import time
import scipy.io as sio
#import itertools

# Input from command line
script, input1 = argv

class Response():

	def __init__(self):
		""" Initialize variables """

		self.input1 = input1

		# Dictionary of methods for processing different input types
		self.inputDict = {'processSingleFile': self.processSingleFile, 'processListFile': self.processListFile, 'processScanner': self.processScanner, 'processSubjID': self.processSubjID}

		# Dictionary of methods for processing different task types
		self.taskDict = {'VAS': self.evalAlcohol, 'Reward': self.evalReward, 'Faces': self.evalFaces, 'RLP': self.evalRLP, 'PSAP': self.evalPSAP, 'aarhus_music': self.evalAarhus, 'breath': self.evalBreath, 'LA_fMRI': self.evalLossAversion}
		self.outputName = {'VAS': 'alcohol', 'Reward': 'reward', 'Faces': 'faces', 'RLP': 'rlp', 'PSAP': 'psap', 'aarhus_music': 'aarhus', 'breath': 'breath', 'LA_fMRI': 'lossAversion'}

		# MRI scanner name scheme
		self.scannerNames = {'p': 'prisma', 'v': 'verio', 'm': 'mmr', 'n': 'mr001'}

		# If not looking in local directory, look here
		self.top = '/rawdata/mr-rh/MRraw/'

		# Regexp for expected behavioral task file name structure
		# self.allMatchTypes = r'^(VAS).*(.txt)$|^(faces[0-9].*(.txt)$|^(reward).*(.txt)$)'
		self.allMatchTypes = r'^(VAS).*(.txt)$|^(HaririFaces[0-9]_dansk).*(.txt)$|^(Hariri_Reward_TC_dansk2_noTrigger).*(.txt)$|^(RLP).*(.txt)$|^(Events).*(.txt)$|^(aarhus-music).*(.txt)$|^(breathing).*(.txt)$|^(LA_fMRI_FourRuns).*(.txt)$'

		# Updated while processing inputs
		self.currTaskType = ''
		self.currSubjID = ''
		self.currSourceFile = ''
		self.currTime = time.strftime('%c')
		self.outFileSuffix = None

		# SPM related structure (.mat file output)
		self.spmStruct = {'names': [], 'onsets': [], 'durations': []}

	def mapInput(self, input1):
		""" Map input to appropriate type """

		fullFilePath = None

		# Inputs starting with "/" assumed to be full path information
		if re.search('^(/).*(.txt)$', input1):
			fullFilePath = input1
			if not path.isfile(fullFilePath):
				raise ValueError('%s does not exist' % fullFilePath)

		# Inputs with .txt suffix and without full path information assumed to be in current directory
		elif re.search('(.txt)$', input1):
			fullFilePath = '/'.join([os.getcwd(), input1])
			if not path.isfile(fullFilePath):
				raise ValueError('%s does not exist' % fullFilePath)

		# If a filename specified as input1
		if fullFilePath:
			try:
				with open(fullFilePath, 'r', encoding='utf-16') as f:
					if f.readline().rstrip() == '*** Header Start ***':
						return ('processSingleFile', fullFilePath)
					else:
						return ('processListFile', fullFilePath)
			except UnicodeError:
				try:
					# PSAP file exception (BOM, not utf-16)
					with open(fullFilePath, 'r') as f:
						if re.search('(Time).*(Event).*(Number)', f.readline().rstrip()):
							return ('processSingleFile', fullFilePath)
				except UnicodeError:
					return ('processListFile', fullFilePath)

		# Check if input matches MRI scanner name
		if input1 in [v for k,v in self.scannerNames.items()]:
			return ('processScanner', input1)

		# If input matches MRI scanner "prefix" (keys in self.scannerNames), assume it is a subject ID
		if re.search(r'^[' + re.escape(''.join(self.scannerNames.keys())) + ']', input1):
			return ('processSubjID', input1)

	def processSingleFile(self, fName):
		""" Process single .txt file input """

		# Pass filename forward
		return [fName]

	def processSubjID(self, subjID):
		""" Process subjID input """

		# All .txt files matching criteria for fMRI behavioral files
		allMatches = []

		# Determine MRI scanner name
		if re.search(r'^[' + re.escape(''.join(self.scannerNames.keys())) + ']', subjID):
			scanner = self.scannerNames[subjID[0]]
		else:
			raise ValueError('No scanner match for %s ' % subjID)

		scannerFolder = ''.join([self.top, scanner])

		# Find all subject IDs that START with "subjID" (can be >1)
		subjIDFolder = [idx for idx in listdir(scannerFolder) if idx.startswith(subjID)]
		if len(subjIDFolder) == 0:
			raise ValueError('No folder match found %s' % subjID)

		fullFolderPath = [''.join([self.top, scanner, '/', idx]) for idx in subjIDFolder]

		# If only one match found, ensure stored as list
		if type(fullFolderPath) is str:
			fullFolderPath = [fullFolderPath]

		# Identify all .txt files matching criteria for fMRI behavioral files among matches to "subjID"
		for folderName in fullFolderPath:
			currFolderList = listdir(folderName)
			currFolderMatches = ['/'.join([folderName, idx]) for idx in currFolderList if re.search(self.allMatchTypes, idx)]
			allMatches += currFolderMatches

		# Return list of .txt files for further processing
		return allMatches

	def processScanner(self, scanner):
		""" Process scanner input """

		# All .txt files matching criteria for fMRI behavioral files
		allMatches = []
		scannerFolder = ''.join([self.top, scanner])

		# List of folders (presumably subject IDs) in scanner folder
		foldersList = listdir(scannerFolder)

		# Identify all .txt files matching criteria for fMRI behavioral files among subjects' folders
		for folderName in foldersList:
			tmpNameFull = '/'.join([scannerFolder, folderName])

			# Make sure tmpNameFull is a folder
			if os.path.isdir(tmpNameFull):
				currFolderList = listdir(tmpNameFull)
				currFolderMatches = ['/'.join([tmpNameFull, idx]) for idx in currFolderList if re.search(self.allMatchTypes, idx)]
				allMatches += currFolderMatches

		# Return list of .txt files for further processing
		return allMatches

	def processListFile(self, fName):
		""" Process list (.txt file) input """

		# List of inputs from .txt file input
		idList = []

		# All .txt files matching criteria for fMRI behavioral files
		allMatches = []
		with open(fName, 'r') as f:
			for line in f:
				l = line.rstrip()
				idList.append(l)

		# Map each line in list. Enables list to contain subject ID, MRI scanner name, or files
		inputInfo = [self.mapInput(idx) for idx in idList]
		for idx in inputInfo:
			allMatches += self.inputDict[idx[0]](idx[1])

		# Return list of .txt files for further processing
		return allMatches

	def taskIdentify(self, fName):
		""" Identify task for given .txt file """

		# Update attribute
		self.currSourceFile = fName
		fileParts = fName.split('/')
		regExpStr = r'^[' + ''.join(self.scannerNames.keys()) + '][0-9][0-9][0-9]'
		# Determine subject ID based on parts of file name
		idMatches = [idx for idx in fileParts if re.search(regExpStr, idx)]

		# Convert list to string and update attribute
		if idMatches:
			self.currSubjID = ''.join(idMatches)
		else:
			self.currSubjID = ''

		# Extract two pieces of information from fMRI behavioral file
		try:
			with open(fName, 'r', encoding='utf-16') as f:
				for line in f:
					if line.startswith('Experiment: '):
						l = line.rstrip()
						l_split = l.split(' ')
						self.currTaskType = ''.join([idx for idx in self.taskDict.keys() if idx in l_split[1]])
		
					# If subject ID has not been determined, use DataFile.Basename from file
					if self.currSubjID == '' and line.startswith('DataFile.Basename'):
						l = line.rstrip()
						l_split = l.split(' ')
						self.currSubjID = l_split[1]
		except UnicodeError:
				try:
					with open(fName, 'r') as f:
						if re.search('(Time).*(Event).*(Number)', f.readline().rstrip()):
							print('here i am')
							self.currTaskType = 'PSAP'
				except UnicodeError:
					pass

		# Return method for identified task type
		return self.taskDict[self.currTaskType]

	def printFile (self, fullFileName):
		""" Print response file """

		with io.open(fullFileName, 'r', encoding = 'utf-16') as f:
			for line in f:
				print(str(line.rstrip()).lstrip())

	def evalLossAversion(self, fullFileName):
		""" Evaluate faces behavioral response file """

		# column names
		colEvents = ['onset', 'duration', 'run_num', 'trial_num', 'stimulus_type', 'target_person', 'gamble_side', 'win_side',
					 'win_amount', 'loss_amount', 'response_time', 'response', 'iti']

		# variables to keep as strings
		keepString = {'Procedure: GambleG[L|R]W[L|R]': 'block_name',
					  'outperson': 'target_person'}

		# variables to keep as numeric
		keepNumeric = {'ITI': 'iti',
					   'WinAmount': 'win_amount',
					   'LossAmount': 'loss_amount',
					   '(Gamble).*(.OnsetTime)': 'gamble_on',
					   '(Fixation).*(.OnsetTime)': 'fix_on',
					   '(GamblePresent).*(RT:)': 'gamble_rt',
					   '(Fixation).*(RT:)': 'fix_rt',
					   '(GamblePresent).*(RESP)': 'gamble_resp',
					   '(Fixation).*(RESP)': 'fix_resp',
					   '^(SelfList[0-9]{1}|Charity[0-9]{1}List).Sample': 'trial_num_current',
					   '^(Level: 2)$': 'level'}

		dfEvent = pd.DataFrame(columns=colEvents)  # event dataframe
		trialCounter = 0
		runCounter = 0
		practice = True

		# Process fMRI behavioral file
		with io.open(fullFileName, 'r', encoding='utf-16') as f:
			for line in f:

				l = str(line.rstrip()).lstrip()

				# signals end of paradigm
				if l == 'Level: 1':
					f.close()
					currTrial['level'] = 1
					dfEvent, trialCounter = self.eventLossAversion(currTrial, dfEvent, trialCounter, runCounter)
					dfEvent.replace(np.nan, 'n/a', inplace=True)
					return {'event': dfEvent, 'summary': None}

				# new trial identifier
				# avoids evaluating practice trials
				if practice is True and re.search('^(Procedure: GambleG[L|R]W[L|R])$', l):
					practice = False
					trialCounter += 1
					runCounter += 1
					currTrial = {'trial_num': trialCounter}  # current trial information

				if re.search('^(\*\*\* LogFrame Start \*\*\*)', l):
					if practice:
						continue
					trialCounter += 1

					dfEvent, trialCounter = self.eventLossAversion(currTrial, dfEvent, trialCounter, runCounter)
					if trialCounter is 0:
						practice = True
						dfEvent = pd.DataFrame(columns=colEvents)  # event dataframe

					currTrial = {'trial_num': trialCounter}

				if practice is True:
					continue

				# pull relevant field for string and numeric values of interest
				strCheck = [v for k, v in keepString.items() if re.search(k, l)]
				numCheck = [v for k, v in keepNumeric.items() if re.search(k, l)]

				if strCheck:
					l_split = l.split(': ')
					if l_split[1] == 'For velgørenhed':
						l_split[1] = 'For velgoerenhed'
					currTrial[''.join(strCheck)] = l_split[1]
				elif numCheck:
					l_split = l.split(': ')
					if len(l_split) == 2:
						currTrial[''.join(numCheck)] = int(l_split[1])
					else:
						currTrial[''.join(numCheck)] = None

		# Return event data frame (summary not written)
		return {'event': dfEvent, 'summary': None}

	def eventLossAversion(self, currTrial, dfEvent, trialCounter, runCounter):
		""" Organize current trial and update dfEvent """

		if all([k in ['trial_num', 'level'] for k in currTrial.keys()]):

			dfEvent = dfEvent.assign(run_num=runCounter)
			dfEvent['onset'] = (dfEvent['onset'] - dfEvent.iloc[0]['onset'])/1000
			dfEvent['response_time'] = [i/1000 if isinstance(i,int) else 'n/a' for i in dfEvent['response_time']]
			dfEvent['iti'] = [i / 1000 if isinstance(i, int) else 'n/a' for i in dfEvent['iti']]
			dfEvent['duration'] = np.append(np.delete(np.array(dfEvent['onset'].diff()), 0), dfEvent['iti'].iloc[-1] / 1000)
			self.outFileSuffix = 'run-' + str(runCounter)
			dfEvent.replace(np.nan, 'n/a', inplace=True)
			if currTrial['level'] is not 1:
				self.writeRespCsv({'event': dfEvent, 'summary': None})
			trialCounter = 0
			return (dfEvent, trialCounter)

		nrep = 2
		if re.search('^(GambleGL)',currTrial['block_name']):
			gamble_side = 'L'
		elif re.search('^(GambleGR)',currTrial['block_name']):
			gamble_side = 'R'

		if re.search('^(GambleG[L|R]WL)',currTrial['block_name']):
			win_side = 'L'
		elif re.search('^(GambleG[L|R]WR)',currTrial['block_name']):
			win_side = 'R'

		if currTrial['gamble_resp'] is None:
			currTrial['gamble_resp'] = 'n/a'
		if currTrial['fix_resp'] is None:
			currTrial['fix_resp'] = 'n/a'

		if currTrial['gamble_rt'] is 0:
			currTrial['gamble_rt'] = 'n/a'
		if currTrial['fix_rt'] is 0:
			currTrial['fix_rt'] = 'n/a'

		# update currEvent
		currEvent = {'onset': [currTrial['gamble_on'], currTrial['fix_on']], 'duration': np.repeat('n/a', nrep),
					 'run_num': np.repeat('n/a', nrep), 'trial_num': np.repeat(currTrial['trial_num_current'], nrep),
					 'stimulus_type': ['gamble', 'fixation'],
					 'target_person': np.repeat(currTrial['target_person'], nrep),
					 'gamble_side': [gamble_side, 'n/a'],
					 'win_side': [win_side, 'n/a'],
					 'win_amount': [currTrial['win_amount'], 'n/a'],
					 'loss_amount': [currTrial['loss_amount'], 'n/a'],
					 'response': [currTrial['gamble_resp'], currTrial['fix_resp']],
					 'response_time': [currTrial['gamble_rt'], currTrial['fix_rt']],
					 'iti': ['n/a', currTrial['iti']]}

		# add currEvent to dfEvent
		dfEvent = dfEvent.append(pd.DataFrame.from_dict(currEvent), ignore_index=True)

		# return updated dfEvent
		return (dfEvent, trialCounter)

	def evalBreath(self, fullFileName):
		""" Evaluate Breathing behavioral response file """

		# column names
		eventInfo = {'onset': [], 'duration': [], 'event_num': [], 'response': []}

		# Process fMRI behavioral file
		with io.open(fullFileName, 'r', encoding='utf-16') as f:
			for line in f:

				l = str(line.rstrip()).lstrip()

				if re.search('KeyboardStimulus.OnsetTime',l):
					l_split = l.split(': ')
					startTime = int(l_split[1])
					eventInfo['onset'].append(0)
					eventInfo['event_num'].append(0)
					eventInfo['response'].append('n/a')
				elif re.search('KeyboardStimulus.OffsetTime',l):
					l_split = l.split(': ')
					endTime = int(l_split[1])
				elif re.search('^(KeyPress[0-9]+RT)',l):
					l_split = l.split(': ')
					eventInfo['onset'].append((int(l_split[1]) - startTime)/1000)
					eventInfo['event_num'].append(len(eventInfo['event_num']))
				elif re.search('^(KeyPress[0-9]+RESP)',l):
					l_split = l.split(': ')
					eventInfo['response'].append(int(l_split[1]))

		# calculate durations
		durations = [np.round(eventInfo['onset'][i+1]-eventInfo['onset'][i],3) for i in range(len(eventInfo['onset'])-1)]
		durations.append(np.round((endTime-startTime)/1000 - eventInfo['onset'][-1],3))
		eventInfo['duration'] = durations

		dfEvent = pd.DataFrame(columns=eventInfo.keys())  # event dataframe
		dfEvent = dfEvent.append(pd.DataFrame.from_dict(eventInfo), ignore_index=True)

		# Return event data frame (summary not written)
		return {'event': dfEvent, 'summary': None}

	def evalPSAP(self, fullFileName):
		""" Evaluate PSAP behavioral response file """
		
		colEvents = ['trial_num', 'option', 'onset', 'duration', 'provocation_time', 'presses']
		dfEvent = pd.DataFrame(columns = colEvents) # event dataframe
		trialCounter = 0
		readLine = False
		
		with io.open(fullFileName,'r') as f:
			for line in f:
				
				l = str(line.strip()).lstrip()
				
				if re.search('Test',l):
					readLine = True
					continue
				
				if readLine:
					if re.search('Begin',l):
						trialCounter  += 1
						currTrial = {'trial_num': trialCounter, 'provocation_time': 'n/a', 'presses': 'n/a'}
						currTrial['onset'] = int(l.split(' ')[0])/1000
						currTrial['option'] = l.split(' ')[-1]
					elif re.search('End',l):
						currTrial['duration'] = int(l.split(' ')[0])/1000 - currTrial['onset']
						dfEvent = dfEvent.append(pd.DataFrame(currTrial, index = [0]), ignore_index = True)
					elif re.search('Provocation',l):
						currTrial['provocation_time'] = int(l.split(' ')[0])/1000
						currTrial['presses'] = int(l.split(' ')[-1])
					else:
						pass
		
		if 'duration' not in currTrial.keys():
			currTrial['duration'] = 'n/a'
			dfEvent = dfEvent.append(pd.DataFrame(currTrial, index = [0]), ignore_index = True)
		
		if 'Opt1' in dfEvent['option']:
			dfEvent.at[dfEvent['option']=='Opt1','option'] = 'monetary'
			dfEvent.at[dfEvent['option']=='Opt2','option'] = 'stealing'
			dfEvent.at[dfEvent['option']=='Opt3','option'] = 'protection'
		else:
			dfEvent.at[dfEvent['option']=='Opt2','option'] = 'monetary'
			dfEvent.at[dfEvent['option']=='Opt3','option'] = 'stealing'
			dfEvent.at[dfEvent['option']=='Opt4','option'] = 'protection'
		
		print(dfEvent)

		# Return event data frame (summary not written)
		return {'event': dfEvent, 'summary': None}
	
	def evalAarhus(self,fullFileName):
		""" Evaluate Aarhus music behavioral response file """
		
		# column names
		colEvents = ['stimulus', 'onset', 'duration', 'file']
		
		# variables to keep as strings
		keepString = {'Procedure': 'music_file'}		
		# variables to keep as numeric
		keepNumeric = {'^(FixOnly.OnsetTime)': 'fix_on', '^(FixMusic[1-4].OnsetTime)': 'music_on'}
		
		paradigmInfo = {'music_file': [], 'fix_on': [], 'music_on': []}
	
		# Process fMRI behavioral file
		with io.open(fullFileName,'r', encoding='utf-16') as f:
			for line in f:
				
				l = str(line.rstrip()).lstrip()
				
				# pull relevant field for string and numeric values of interest
				strCheck = [v for k,v in keepString.items() if re.search(k,l)]
				numCheck = [v for k,v in keepNumeric.items() if re.search(k,l)]
				
				if strCheck:
					paradigmInfo[''.join(strCheck)].append(l.split(': ')[1])
				if numCheck:
					paradigmInfo[''.join(numCheck)].append(int(l.split(': ')[1]))
		
		# create variables for df
		stimulus = ['fixation','music','fixation','music','fixation','music','fixation','music','fixation']
		files = ['n/a',paradigmInfo['music_file'][0],'n/a',paradigmInfo['music_file'][1],'n/a',paradigmInfo['music_file'][2],'n/a',paradigmInfo['music_file'][3],'n/a']
		allonsets = paradigmInfo['fix_on'] + paradigmInfo['music_on']
		allonsets.sort()
		onsets = [(i-allonsets[0])/1000 for i in allonsets]
		durations = [onsets[i]-onsets[i-1] for i in range(1,len(onsets))]
		durations.append(10)
		
		currEvent = {'stimulus': stimulus, 'onset': onsets, 'duration': durations, 'file': files}
		dfEvent = pd.DataFrame(columns = colEvents) # event dataframe
		dfEvent = dfEvent.append(pd.DataFrame.from_dict(currEvent), ignore_index = True)
		
		# Return event data frame (summary not written)
		return {'event': dfEvent, 'summary': None}
	
	def evalFaces(self, fullFileName):
		""" Evaluate faces behavioral response file """
		
		# list of stimuli
		stimulus_list = ['Shapes', 'Fear', 'Neutral', 'Angry', 'Surprise']
		
		# column names
		colEvents = ['block_name', 'block_num', 'block_subtype', 'trial_num', 'stimulus_type', 'onset', 'duration', 'response_time', 'response', 'correct_response', 'stimulus_file']
		
		# variables to keep as strings
		keepString = {'TrialCondition': 'block_name', 'Procedure: (' + '|'.join(stimulus_list) + ')(BlockProc|FacesBlock)': 'block_name', 'Procedure: (' + '|'.join(stimulus_list) + ')(Trial|Faces)Proc': 'stimulus_type', 'Experiment': 'run_type', 'Stimulus': 'stimulus_file'}
		
		# variables to keep as numeric
		keepNumeric = {'(' + '|'.join(stimulus_list) + ')(TrialProbe|FacesProcProbe)(.OnsetTime)': 'stim_on', '^(' + '|'.join(stimulus_list) + ')(TrialProbe|FacesProcProbe)(.RT)': 'rt', '^(' + '|'.join(stimulus_list) + ')(TrialProbe|FacesProcProbe)(.RESP)': 'resp', '^(' + '|'.join(stimulus_list) + ')(TrialProbe|FacesProcProbe)(.CRESP)': 'cresp', '^(' + '|'.join(stimulus_list) + ')(TrialFixation|FacesProcFix)(.OnsetTime)': 'fix_on', '^(FacesRunBlockList)': 'block_num', '^(Match)(Shapes|Faces)(.OnsetTime)': 'onset'}
		
		dfEvent = pd.DataFrame(columns = colEvents) # event dataframe
		trialCounter = 0
		currTrial = {'trial_num': trialCounter} # current trial information
	
		# Process fMRI behavioral file
		with io.open(fullFileName,'r', encoding='utf-16') as f:
			for line in f:
				
				l = str(line.rstrip()).lstrip()
				
				# signals end of paradigm
				if l=='Procedure: FacesRunProc':
					f.close()
					dfEvent = self.eventFaces({}, dfEvent, trialCounter)
					dfEvent.replace(np.nan,'n/a',inplace=True)
					return {'event': dfEvent, 'summary': None}
				
				# new trial identifier
				if re.search('^(\*\*\* LogFrame Start \*\*\*)',l):
					trialCounter += 1
					if trialCounter > 1:
						dfEvent, trialCounter = self.eventFaces(currTrial, dfEvent, trialCounter)
					
					currTrial = {'trial_num': trialCounter}
				
				# pull relevant field for string and numeric values of interest
				strCheck = [v for k,v in keepString.items() if re.search(k,l)]
				numCheck = [v for k,v in keepNumeric.items() if re.search(k,l)]
				
				if strCheck:
					l_split = l.split(': ')
					currTrial[''.join(strCheck)] = l_split[1]
				elif numCheck:
					l_split = l.split(': ')
					if len(l_split) == 2:
						currTrial[''.join(numCheck)] = int(l_split[1])
						if ''.join(numCheck) == 'rt' and currTrial[''.join(numCheck)] == 0:
							currTrial[''.join(numCheck)] = None
					else:
						currTrial[''.join(numCheck)] = None
		
		# Return event data frame (summary not written)
		return {'event': dfEvent, 'summary': None}
	
	def eventFaces(self, currTrial, dfEvent, trialCounter):
		""" Organize current trial and update dfEvent """
		
		stimulus_list = ['Shapes', 'Fear', 'Neutral', 'Angry', 'Surprise']
		
		# processing end of paradigm
		if len(currTrial) == 0:
			
			# convert onsets to seconds
			dfEvent['onset'] = (dfEvent['onset']-dfEvent.loc[0,'onset'])/1000
			
			# calculate durations
			for elem in dfEvent.index:
				if elem < len(dfEvent.index)-1:
					dfEvent.loc[elem,'duration'] = dfEvent.loc[elem+1,'onset']-dfEvent.loc[elem,'onset']
				else:
					dfEvent.loc[elem,'duration'] = 2
			
			# return updated dfEvent (trialCounter need not be returned)
			return(dfEvent)		
		
		nrep = 2 # faces trials comprise four events (stimulus, fixation)
		
		# "match shapes" and "match faces" events handled differently
		if all([i in currTrial.keys() for i in ['trial_num', 'block_name', 'block_num', 'onset']]):
			#currTrial['outcome'] = 'Instruction'
			if re.search('Shapes', currTrial['block_name']):
				currTrial['stimulus_type'] = 'MatchShapes'
			else:
				currTrial['stimulus_type'] = 'MatchFaces'
			
			dfEvent = dfEvent.append(pd.DataFrame(currTrial, index = [0]), ignore_index = True)
			dfEvent.loc[dfEvent['block_name'].isnull(), 'block_name'] = currTrial['block_name']
			dfEvent.loc[dfEvent['block_num'].isnull(), 'block_num'] = currTrial['block_num']
			dfEvent.loc[dfEvent.index[-1], 'trial_num'] = None
			
			# update row order; guess number/press finger comes after block trials in paradigm file
			currBlockIdx = dfEvent[dfEvent['block_num'] == currTrial['block_num']].index
			prevBlockIdx = dfEvent[dfEvent['block_num'] != currTrial['block_num']].index
			dfEvent.loc[currBlockIdx[-1],'block_name'] = 'Instruction'
			dfEvent = dfEvent.reindex(list(prevBlockIdx) + [currBlockIdx[-1]] + list(currBlockIdx[:-1]))
			dfEvent = dfEvent.reset_index(drop=True)
			trialCounter -= 1
			
		else:
			
			currOnsets = [currTrial['stim_on'], currTrial['fix_on']]
			currTrial['block_subtype'] = ''.join([idx for idx in stimulus_list if re.search(idx,currTrial['stimulus_type'])])
			
			# update currEvent
			currEvent = {'trial_num': np.repeat(currTrial['trial_num'],nrep), 'block_name': np.repeat(currTrial['block_name'],nrep), 'block_subtype': np.repeat(currTrial['block_subtype'],nrep), 'stimulus_type': ['stimulus', 'fixation'], 'onset': currOnsets, 'duration': np.repeat('n/a',nrep), 'response_time': [currTrial['rt'], 'n/a'], 'response': [currTrial['resp'], 'n/a'], 'correct_response': [currTrial['cresp'], 'n/a'], 'stimulus_file': currTrial['stimulus_file']}
		
			# add currEvent to dfEvent
			dfEvent = dfEvent.append(pd.DataFrame.from_dict(currEvent), ignore_index = True)
		
		# return updated dfEvent
		return(dfEvent, trialCounter)
	
	def evalFaces_old(self, fullFileName):
		""" Evaluate faces behavioral response file """

		# Update attribute to incorporate version of faces task (i.e., 1,2,3, or 4)
		self.outputName['Faces'] = ''.join(['faces', self.currSourceFile.split('HaririFaces')[1][0]])

		colEvents = ['type', 'subtype', 'imgName', 'rt', 'resp', 'cresp', 'acc', 'omitted']
		colSummary = ['RT_corrOnly', 'RT_all', 'accuracy', 'correct', 'incorrect', 'omitted']
		rowSummary = ['shapes', 'fear', 'angry', 'neutral', 'surprise', 'faces']

		dfEvent = pd.DataFrame(columns = colEvents)
		dfSummary = pd.DataFrame(index = rowSummary, columns = colSummary)

		# New trial information identifiers.
		trialType = {'Procedure: ShapesTrialProc': 'shapes', 'Procedure: FearFacesProc': 'fear', 'Procedure: NeutralFacesProc': 'neutral', 'Procedure: AngryFacesProc': 'angry', 'Procedure: SurpriseFacesProc': 'surprise'}

		# Process fMRI behavioral file
		with io.open(fullFileName,'r', encoding='utf-16') as f:
		    for line in f:

		        l = str(line.rstrip()).lstrip()

		        # trial subtype
		        if l in trialType.keys():
		            dfEvent.loc[len(dfEvent)+1] = pd.Series({'subtype': trialType[l]})

		        # trial type
		        if re.search('^(TrialCondition)', l):
		            l_split = l.split(' ')
		            dfEvent.at[len(dfEvent), 'type'] = l_split[1]

		        # accuracy
		        if re.search('.*(Probe).*(ACC).*', l):
		            l_split = l.split(' ')
		            dfEvent.at[len(dfEvent), 'acc'] = int(l_split[1])

		        # reaction time
		        if re.search('.*(Probe).*(\.RT:).*', l):
		            l_split = l.split(' ')
		            if len(l_split) == 2:
		                dfEvent.at[len(dfEvent), 'rt']  = int(l_split[1])
		                dfEvent.at[len(dfEvent), 'omitted']  = False
		            else:
		                dfEvent.at[len(dfEvent), 'omitted']  = True

		        # actual response
		        if re.search('.*(Probe).*(\.RESP).*', l):
		            l_split = l.split(' ')
		            if len(l_split) == 2:
		                dfEvent.at[len(dfEvent), 'resp']  = int(l_split[1])

		        # correct response
		        if re.search('.*(Probe).*(\.CRESP).*', l):
		            l_split = l.split(' ')
		            dfEvent.at[len(dfEvent), 'cresp']  = int(l_split[1])

		        # stimulus (image file name)
		        if re.search('^(Stimulus)', l):
		            l_split = l.split(' ')
		            dfEvent.at[len(dfEvent), 'imgName']  = l_split[1]

		# Compute summary measures
		for k,v in trialType.items():
			dfSummary.at[v, 'RT_corrOnly'] = np.mean(dfEvent.loc[(dfEvent['subtype'] == v) & dfEvent['acc'] == 1, 'rt'])
			dfSummary.at[v, 'RT_all'] = np.mean(dfEvent.loc[dfEvent['subtype'] == v, 'rt'])
			dfSummary.at[v, 'accuracy'] = np.mean(dfEvent.loc[dfEvent['subtype'] == v, 'acc'])
			dfSummary.at[v, 'correct'] = np.sum(dfEvent.loc[dfEvent['subtype'] == v, 'acc'])
			dfSummary.at[v, 'omitted'] = np.sum(dfEvent.loc[dfEvent['subtype'] == v, 'omitted'])
			dfSummary.at[v, 'incorrect'] = np.sum(dfEvent['subtype'] == v) - dfSummary.at[v, 'correct'] - dfSummary.at[v, 'omitted']

		dfSummary.at['faces', 'RT_corrOnly'] = np.mean(dfEvent.loc[(dfEvent['type'] == 'faces') & dfEvent['acc'] == 1, 'rt'])
		dfSummary.at['faces', 'RT_all'] = np.mean(dfEvent.loc[dfEvent['type'] == 'faces', 'rt'])
		dfSummary.at['faces', 'accuracy'] = np.mean(dfEvent.loc[dfEvent['type'] == 'faces', 'acc'])
		dfSummary.at['faces', 'correct'] = np.sum(dfEvent.loc[dfEvent['type'] == 'faces', 'acc'])
		dfSummary.at['faces', 'omitted'] = np.sum(dfEvent.loc[dfEvent['type'] == 'faces', 'omitted'])
		dfSummary.at['faces', 'incorrect'] = np.sum(dfEvent['type'] == 'faces') - dfSummary.at['faces', 'correct'] - dfSummary.at['faces', 'omitted']

		# Clean up fractions
		dfSummary['RT_corrOnly'] = [float("{0:.3f}".format(idx)) for idx in dfSummary['RT_corrOnly']]
		dfSummary['RT_all'] = [float("{0:.3f}".format(idx)) for idx in dfSummary['RT_all']]
		dfSummary['accuracy'] = [float("{0:.3f}".format(idx)) for idx in dfSummary['accuracy']]

		# Transpose to match how Peter reads it into database
		dfSummary = dfSummary.transpose()

		# Return even and summary data frames
		return {'event': dfEvent, 'summary': dfSummary}

	def evalReward(self,fullFileName):
		""" Evaluate reward behavioral response file """
		
		# column names
		colEvents = ['block_name', 'block_num', 'trial_num', 'stimulus_type', 'onset', 'duration', 'response_time',
					 'response', 'high_num', 'low_num', 'num_shown', 'outcome', 'feedback']
		
		# variables to keep as strings
		keepString = {'TrialCondition': 'subtype', 'lowNum': 'low_num', 'highNum': 'high_num', '(Reward|Loss|Control)BlockProc': 'block_name'}
		
		# variables to keep as numeric
		keepNumeric = {'^(GamStim.OnsetTime|ControlStim.OnsetTime)': 'guess_on', '^(GamStim.RT|ControlStim.RT)': 'rt',
					   '^(GamStim.RESP|ControlStim.RESP)': 'resp',
					   '^(Feedback[RL]{1}.OnsetTime|ControlFeedbackStar.OnsetTime)': 'num_on',
					   '^(Feedback(Up|Down)Arrow.OnsetTime|ControlFeedbackCircle.OnsetTime)': 'feed_on',
					   'GuessingFixation.OnsetTime': 'fix_on', '^(GuessingRunBlockList)': 'block_num',
					   '^(GuessNumber.OnsetTime|PressButton.OnsetTime)': 'onset',
					   '^(GamStim.OnsetToOnsetTime|ControlStim.OnsetToOnsetTime)': 'onset2onset'}

		# feedback dictionary
		feedbackDict = {'Up': 'up_arrow', 'Down': 'down_arrow', 'Circle': 'circle'}
		
		dfEvent = pd.DataFrame(columns = colEvents) # event dataframe
		trialCounter = 0
		currTrial = {'trial_num': trialCounter} # current trial information
		allTrials = [] # work around for .txt file missing some onset times
	
		# Process fMRI behavioral file
		with io.open(fullFileName,'r', encoding='utf-16') as f:
			for line in f:
				
				l = str(line.rstrip()).lstrip()

				# signals end of paradigm
				if l=='Procedure: GuesingRunProc':
					f.close()
					if len(allTrials) > 0:
						dfEvent = self.eventReward2(allTrials,dfEvent,trialCounter)
					else:
						dfEvent = self.eventReward({}, dfEvent, trialCounter)
					dfEvent.replace(np.nan, 'n/a', inplace=True)
					return {'event': dfEvent, 'summary': None}
				
				# new trial identifier
				if re.search('^(\*\*\* LogFrame Start \*\*\*)',l):
					trialCounter += 1
					if trialCounter > 1:
						if any(['guess_on' and not 'num_on' in currTrial.keys(), 'block_name' and not 'onset' in currTrial.keys()]) and len(currTrial.keys())>1:
							allTrials.append(currTrial)
						else:
							dfEvent, trialCounter = self.eventReward(currTrial, dfEvent, trialCounter)
					
					currTrial = {'trial_num': trialCounter}
				
				# pull relevant field for string and numeric values of interest
				strCheck = [v for k,v in keepString.items() if re.search(k,l)]
				numCheck = [v for k,v in keepNumeric.items() if re.search(k,l)]
				
				if strCheck:
					l_split = l.split(': ')
					currTrial[''.join(strCheck)] = l_split[1]
					if re.search('(Reward|Loss|Control)BlockProc',l_split[1]):
						currTrial[''.join(strCheck)] = l_split[1].split('BlockProc')[0]
				elif numCheck:
					l_split = l.split(': ')
					if len(l_split) == 2:
						currTrial[''.join(numCheck)] = int(l_split[1])
						if ''.join(numCheck) == 'rt' and currTrial[''.join(numCheck)] == 0:
							currTrial[''.join(numCheck)] = None
						if ''.join(numCheck) == 'feed_on':
							currTrial['feedback'] = ''.join([v for k,v in feedbackDict.items() if re.search(k,l_split[0])])
					else:
						currTrial[''.join(numCheck)] = None
	
	def eventReward(self, currTrial, dfEvent, trialCounter):
		""" Organize current trial and update dfEvent """
		
		# processing end of paradigm
		if len(currTrial) == 0:
			
			# convert onsets to seconds
			dfEvent['onset'] = (dfEvent['onset']-dfEvent.loc[0,'onset'])/1000
			
			# calculate durations
			for elem in dfEvent.index:
				if elem < len(dfEvent.index)-1:
					dfEvent.loc[elem,'duration'] = dfEvent.loc[elem+1,'onset']-dfEvent.loc[elem,'onset']
				else:
					dfEvent.loc[elem,'duration'] = 3
			
			# return updated dfEvent (trialCounter need not be returned)
			return(dfEvent)		
		
		nrep = 4 # reward trials comprise four events (guess, number, feedback, fixation)
		
		# "guess number" and "press finger" events handled differently
		if all([i in currTrial.keys() for i in ['trial_num', 'block_name', 'block_num', 'onset']]):
			
			if currTrial['block_name'] in ['Reward','Loss']:
				currTrial['stimulus_type'] = 'GuessNumber'
			elif currTrial['block_name'] == 'Control':
				currTrial['stimulus_type'] = 'PressFinger'
			
			dfEvent = dfEvent.append(pd.DataFrame(currTrial, index = [0]), ignore_index = True)
			dfEvent.loc[dfEvent['block_name']=='n/a', 'block_name'] = currTrial['block_name']
			#print(dfEvent)
			#print(currTrial['block_num'])
			dfEvent.loc[dfEvent['block_num'].isna(), 'block_num'] = currTrial['block_num']
			print(dfEvent)
			dfEvent.loc[dfEvent.index[-1], 'trial_num'] = 'n/a'
			
			# update row order; guess number/press finger comes after block trials in paradigm file
			currBlockIdx = dfEvent[dfEvent['block_num'] == currTrial['block_num']].index
			prevBlockIdx = dfEvent[dfEvent['block_num'] != currTrial['block_num']].index
			dfEvent.loc[currBlockIdx[-1],'block_name'] = 'Instruction'
			#print(list(prevBlockIdx) + [currBlockIdx[-1]] + list(currBlockIdx[:-1]))
			dfEvent = dfEvent.reindex(list(prevBlockIdx) + [currBlockIdx[-1]] + list(currBlockIdx[:-1]))
			dfEvent = dfEvent.reset_index(drop=True)
			trialCounter -= 1
			
		else:
			
			currOnsets = [currTrial['guess_on'], currTrial['num_on'], currTrial['feed_on'], currTrial['fix_on']]
			
			if currTrial['subtype'] == 'Reward' and currTrial['resp']:
				currTrial['num_shown'] = [currTrial['low_num'], currTrial['high_num']][currTrial['resp']==3]
			elif currTrial['subtype'] == 'Loss' and currTrial['resp']:
				currTrial['num_shown'] = [currTrial['low_num'], currTrial['high_num']][currTrial['resp']==2]
			elif currTrial['subtype'] == 'Control':
				currTrial['high_num'] = None
				currTrial['low_num'] = None
				if currTrial['resp']:
					currTrial['num_shown'] = '*'
				else:
					currTrial['num_shown'] = '---'
					currTrial['feedback'] = '---'
			elif not currTrial['resp']:
				currTrial['num_shown'] = '---'
				currTrial['feedback'] = '---'
			
			if currTrial['subtype'] in ['Reward','Loss']:
				currTrial['trial_type'] = 'guess'
			else:
				currTrial['trial_type'] = 'press'
			
			if currTrial['rt'] is not None:
				currTrial['rt'] = currTrial['rt']/1000
			
			# update currEvent
			currEvent = {'trial_num': np.repeat(currTrial['trial_num'],nrep), 'block_name': np.repeat('n/a',nrep), 'stimulus_type': ['guess', 'number', 'feedback', 'fixation'], 'onset': currOnsets, 'duration': np.repeat('n/a',nrep), 'response_time': [currTrial['rt'], 'n/a', 'n/a', 'n/a'], 'response': [currTrial['resp'], 'n/a', 'n/a', 'n/a'], 'num_shown': ['n/a', currTrial['num_shown'], 'n/a', 'n/a'], 'high_num': ['n/a', currTrial['high_num'], 'n/a', 'n/a'], 'low_num': ['n/a', currTrial['low_num'], 'n/a', 'n/a'], 'outcome': ['n/a', 'n/a', currTrial['subtype'], 'n/a'], 'feedback': ['n/a', 'n/a', currTrial['feedback'], 'n/a']}
		
			# add currEvent to dfEvent
			dfEvent = dfEvent.append(pd.DataFrame.from_dict(currEvent), ignore_index = True)
		
		# return updated dfEvent
		return(dfEvent, trialCounter)
	
	def eventReward2(self, allTrials, dfEvent, trialCounter):
		""" Organizes trials with limited paradigm timing information """
		""" MR001 CBS-entrepreneurship project e-prime behavioral .txt files contain fewer event time stamps """

		#colEvents = ['onset', 'duration', 'block_name', 'block_num', 'trial_num', 'stimulus_type', 'response_time', 'response', 'high_num', 'low_num', 'num_shown', 'outcome', 'feedback']

		#dfEvent = pd.DataFrame(columns=colEvents)  # event dataframe

		instr_dur    = 3 # instructions duration fixed at 3 s
		nrep         = 4 # each trial has four parts
		trialCounter = 0
		blockCounter = 1

		for i in range(len(allTrials) - 1):
			if i == 0:
				currBlock = {'onset': [0], 'duration': [instr_dur], 'block_name': ['Instruction'],
							 'block_num': [blockCounter], 'trial_num': ['n/a'], 'stimulus_type': ['n/a'],
							 'response_time': ['n/a'], 'response': ['n/a'], 'high_num': ['n/a'], 'low_num': ['n/a'],
							 'num_shown': ['n/a'], 'outcome': ['n/a'], 'feedback': ['n/a']}
				overallStartTime = allTrials[i]['guess_on'] - instr_dur * 1000

			if 'subtype' in allTrials[i].keys():
				trialCounter += 1
				currGuessOn = (allTrials[i]['guess_on'] - overallStartTime) / 1000
				currGuessDur = (allTrials[i]['onset2onset']) / 1000
				currNumbrOn = currGuessOn + (allTrials[i]['onset2onset'] / 1000)
				currNumbrDur = 0.5
				currFeedbOn = currNumbrOn + currNumbrDur
				currFeedbDur = 0.5
				currFixatOn = currFeedbOn + currFeedbDur

				if 'subtype' in allTrials[i + 1].keys():
					nextGuessOn = (allTrials[i + 1]['guess_on'] - overallStartTime) / 1000
					currFixatDur = nextGuessOn - currGuessOn - currGuessDur - currNumbrDur - currFeedbDur
				elif i + 2 < len(allTrials) - 1:
					nextGuessOn = (allTrials[i + 2]['guess_on'] - overallStartTime) / 1000
					currFixatDur = nextGuessOn - currGuessOn - currGuessDur - currNumbrDur - currFeedbDur - instr_dur

				currBlock['onset'] = currBlock['onset'] + [currGuessOn, currNumbrOn, currFeedbOn, currFixatOn]
				currBlock['duration'] = currBlock['duration'] + [currGuessDur, currNumbrDur, currFeedbDur, currFixatDur]
				currBlock['block_name'] = currBlock['block_name'] + ['n/a'] * nrep
				currBlock['block_num'] = currBlock['block_num'] + [blockCounter] * nrep
				currBlock['trial_num'] = currBlock['trial_num'] + [trialCounter] * nrep
				currBlock['stimulus_type'] = currBlock['stimulus_type'] + ['guess', 'number', 'feedback', 'fixation']
				if allTrials[i]['rt'] is not None:
					currBlock['response_time'] = currBlock['response_time'] + [allTrials[i]['rt'] / 1000] + ['n/a'] * (
								nrep - 1)
					currBlock['response'] = currBlock['response'] + [allTrials[i]['resp']] + ['n/a'] * (nrep - 1)
				else:
					currBlock['response_time'] = currBlock['response_time'] + ['n/a'] * nrep
					currBlock['response'] = currBlock['response'] + ['n/a'] * nrep

				if allTrials[i]['subtype'] in ['Reward', 'Loss']:
					currBlock['high_num'] = currBlock['high_num'] + ['n/a'] + [allTrials[i]['high_num']] + ['n/a'] * 2
					currBlock['low_num'] = currBlock['low_num'] + ['n/a'] + [allTrials[i]['low_num']] + ['n/a'] * 2
					if allTrials[i]['subtype'] == 'Reward' and allTrials[i]['resp']:
						num_shown = [allTrials[i]['low_num'], allTrials[i]['high_num']][allTrials[i]['resp'] == 3]
						currBlock['num_shown'] = currBlock['num_shown'] + ['n/a'] + [num_shown] + ['n/a'] * 2
						currBlock['outcome'] = currBlock['outcome'] + ['n/a'] * 2 + ['Reward'] + ['n/a']
						currBlock['feedback'] = currBlock['feedback'] + ['n/a'] * 2 + ['up_arrow'] + ['n/a']
					elif allTrials[i]['subtype'] == 'Loss' and allTrials[i]['resp']:
						num_shown = [allTrials[i]['low_num'], allTrials[i]['high_num']][allTrials[i]['resp'] == 2]
						currBlock['num_shown'] = currBlock['num_shown'] + ['n/a'] + [num_shown] + ['n/a'] * 2
						currBlock['outcome'] = currBlock['outcome'] + ['n/a'] * 2 + ['Loss'] + ['n/a']
						currBlock['feedback'] = currBlock['feedback'] + ['n/a'] * 2 + ['down_arrow'] + ['n/a']
					else:
						currBlock['num_shown'] = currBlock['num_shown'] + ['n/a'] + ['---'] + ['n/a'] * 2
						currBlock['outcome'] = currBlock['outcome'] + ['n/a'] * 2 + ['---'] + ['n/a']
						currBlock['feedback'] = currBlock['feedback'] + ['n/a'] * 2 + ['---'] + ['n/a']
				elif allTrials[i]['subtype'] == 'Control':
					currBlock['high_num'] = currBlock['high_num'] + ['n/a'] * nrep
					currBlock['low_num'] = currBlock['low_num'] + ['n/a'] * nrep
					if allTrials[i]['resp']:
						currBlock['num_shown'] = currBlock['num_shown'] + ['n/a'] + ['*'] + ['n/a'] * 2
						currBlock['outcome'] = currBlock['outcome'] + ['n/a'] * 2 + ['Control'] + ['n/a']
						currBlock['feedback'] = currBlock['feedback'] + ['n/a'] * 2 + ['circle'] + ['n/a']
					else:
						currBlock['num_shown'] = currBlock['num_shown'] + ['n/a'] + ['---'] + ['n/a'] * 2
						currBlock['outcome'] = currBlock['outcome'] + ['n/a'] * 2 + ['---'] + ['n/a']
						currBlock['feedback'] = currBlock['feedback'] + ['n/a'] * 2 + ['---'] + ['n/a']

			elif 'block_name' in allTrials[i].keys():
				currBlock['block_name'] = list(
					map(lambda x: x.replace('n/a', allTrials[i]['block_name']), currBlock['block_name']))
				if allTrials[i]['block_name'] in ['Reward', 'Loss']:
					currBlock['block_name'] = list(
						map(lambda x: x.replace('n/a', 'GuessNumber'), currBlock['block_name']))
				else:
					currBlock['block_name'] = list(
						map(lambda x: x.replace('n/a', 'PressButton'), currBlock['block_name']))

				if i < len(allTrials) - 1:
					currBlock['onset'].append(currBlock['onset'][-1] + currBlock['duration'][-1])
					currBlock['duration'].append(instr_dur)
					currBlock['block_name'].append('Instruction')
					currBlock['block_num'].append(currBlock['block_num'][-1] + 1)
					blockCounter += 1
					currBlock['trial_num'].append('n/a')
					currBlock['stimulus_type'].append('n/a')
					currBlock['response_time'].append('n/a')
					currBlock['response'].append('n/a')
					currBlock['high_num'].append('n/a')
					currBlock['low_num'].append('n/a')
					currBlock['num_shown'].append('n/a')
					currBlock['outcome'].append('n/a')
					currBlock['feedback'].append('n/a')

		dfEvent = pd.DataFrame(currBlock)
		return(dfEvent)
	
	def evalReward_old(self, fullFileName):
		""" Evaluate reward behavioral response file """
		""" BIDS-related update (July 2022) highlighted need to change format of reward event file. Old function kept in case need be used. PMF 06-07-2022 """

		colEvents = ['type', 'subtype', 'RT', 'resp', 'win_loss', 'omitted', 'pressFixation']
		colSummary = ['RT', 'Omit', 'Omit_frac', 'Wins', 'Losses']
		rowSummary = ['Reward', 'NoReward', 'Control', 'Total']

		dfEvent = pd.DataFrame(columns = colEvents)
		dfSummary = pd.DataFrame(index = rowSummary, columns = colSummary)

		# Process fMRI behavioral file
		with io.open(fullFileName,'r', encoding='utf-16') as f:
			for line in f:
				l = str(line.rstrip()).lstrip()

				# Assign subtype
				if re.search('^TrialCondition', l):
					l_split = l.split(' ')
					dfEvent.loc[len(dfEvent)+1] = pd.Series({'subtype': l_split[1].lower()})

				# Determine response
				if re.search('GamStim.RESP|ControlStim.RESP', l):
					l_split = l.split(' ')
					if len(l_split) == 2:
						dfEvent.loc[len(dfEvent), 'resp'] = int(l_split[1])
						dfEvent.loc[len(dfEvent), 'omitted'] = False
					else:
						dfEvent.loc[len(dfEvent), 'omitted'] = True

				# Reaction time
				if re.search('(GamStim.RT\:)|(ControlStim.RT\:)', l):
					l_split = l.split(' ')
					if len(l_split) == 2 and int(l_split[1]) > 0:
						dfEvent.loc[len(dfEvent), 'RT'] = int(l_split[1])
					else:
						dfEvent.loc[len(dfEvent), 'RT'] = None

					if re.search('GuessingFixation.RESP',l):
						l_split = l.split(' ')
						if len(l_split) == 2:
							dfEvent.loc[len(dfEvent), 'pressFixation'] = 1

				# Determine type (due to .txt file structure, cannot be determined until after block is completed)
				if re.search('^(Procedure).*RewardBlockProc', l):
					dfEvent.iloc[-5:,dfEvent.columns.get_loc('type')] = 'Reward'
				elif re.search('^(Procedure).*LossBlockProc', l):
					dfEvent.iloc[-5:,dfEvent.columns.get_loc('type')] = 'NoReward'
				elif re.search('^(Procedure).*ControlBlockProc', l):
					dfEvent.iloc[-5:,dfEvent.columns.get_loc('type')] = 'Control'

		# Money won or lost
		dfEvent.loc[(dfEvent['subtype'] == 'reward') & (~dfEvent['resp'].isnull()), 'win_loss'] = 'win'
		dfEvent.loc[(dfEvent['subtype'] == 'loss'), 'win_loss'] = 'loss'
		dfEvent.loc[((dfEvent['subtype'] == 'reward') & (dfEvent['resp'].isnull())), 'win_loss'] = 'loss'

		# Summary measures
		for t in dfEvent.type.unique():
			dfSummary.at[t, 'RT'] = np.mean(dfEvent.loc[dfEvent['type'] == t, 'RT'])
			dfSummary.at[t, 'Omit'] = np.sum(dfEvent.loc[dfEvent['type'] == t, 'omitted'])
			dfSummary.at[t, 'Omit_frac'] = dfSummary.loc[t, 'Omit']/len(dfEvent.loc[dfEvent['type'] == t])
			dfSummary.at[t, 'Wins'] = np.sum(dfEvent.loc[dfEvent['type'] == t, 'win_loss'] == 'win')
			dfSummary.at[t, 'Losses'] = np.sum(dfEvent.loc[dfEvent['type'] == t, 'win_loss'] == 'loss')

		dfSummary.at['Total', 'RT'] = np.mean(dfEvent['RT'])
		dfSummary.at['Total', 'Omit'] = np.sum(dfEvent['omitted'])
		dfSummary.at['Total', 'Omit_frac'] = dfSummary.at['Total', 'Omit']/len(dfEvent)
		dfSummary.at['Total', 'Wins'] = np.sum(dfEvent['win_loss'] == 'win')
		dfSummary.at['Total', 'Losses'] = np.sum(dfEvent['win_loss'] == 'loss')

		# Clean fractions
		dfSummary['RT'] = [float("{0:.3f}".format(idx)) for idx in dfSummary['RT']]
		dfSummary['Omit_frac'] = [float("{0:.3f}".format(idx)) for idx in dfSummary['Omit_frac']]

		# Transpose to match how Peter reads it into database
		dfSummary = dfSummary.transpose()

		# Return even and summary data frames
		return {'event': dfEvent, 'summary': dfSummary}

	def evalAlcohol(self, fullFileName):
		""" Evaluate alcohol craving response file """

		colEvents = ['type', 'subtype', 'fileName', 'duration', 'onsetFromStart', 'rating', 'ratingOmit']
		colSummary = ['ratingDuration', 'ratingAvg', 'ratingOmit']
		rowSummary = ['neutral', 'wine', 'schnapps', 'beer', 'allAlcohol']
		allAlcohol = ['wine', 'schnapps', 'beer']

		# New trial information identifiers.
		trialType = {'Procedure: ImgNeutralPresent': ('neutral', 'neutral'), 'Procedure: VasOnlyExample': ('rating', 'rating'), 'Procedure: ImgWinePresent': ('alcohol', 'wine'), 'Procedure: ImgSchnappsPresent': ('alcohol', 'schnapps'), 'Procedure: ImgBeerPresent': ('alcohol', 'beer')}

		# Onset times (absolute)
		times = {'old': ('', 0), 'new': ('', 0)}

		# Clock at task start (absolute)
		startTime = None

		dfEvent = pd.DataFrame(columns = colEvents)
		dfSummary = pd.DataFrame(index = rowSummary, columns = colSummary)

		# Process fMRI behavioral file
		with io.open(fullFileName,'r', encoding='utf-16') as f:
			for line in f:
				l = str(line.rstrip()).lstrip()

				# Assign subtype
				if l in trialType.keys():
					dfEvent.loc[len(dfEvent)+1] = pd.Series({'type': trialType[l][0], 'subtype': trialType[l][1]})
					if trialType[l][0] == 'rating':
						dfEvent.at[len(dfEvent), 'fileName'] = dfEvent.at[len(dfEvent)-1, 'subtype']

				# stimulus (image file name)
				if re.search('^filename\:', l):
					l_split = l.split(' ')
					dfEvent.loc[len(dfEvent), 'fileName'] = l_split[1]

				# Onset time (absolute)
				if re.search('.*(OnsetTime\:)', l):
					l_split = l.split(' ')
					l_split[1] = int(l_split[1])
					if startTime is None:
						startTime = l_split[1]

					times['old'] = times['new']
					times['new'] = (l_split[0], l_split[1])

					# Calculate onset time (relative)
					if re.search('.*(Image\.OnsetTime)|VasSlide\.OnsetTime', times['new'][0]):
						dfEvent.at[len(dfEvent), 'onsetFromStart'] = float(l_split[1] - startTime)/1000

					# Stimulus duration
					if re.search('.*(Image\.OnsetTime)', times['old'][0]):
						diff = (times['new'][1] - times['old'][1])
						dfEvent.at[len(dfEvent)-1, 'duration'] = float(diff)/1000

					# Rating scale duration
					if re.search('VasSlide\.OnsetTime', times['old'][0]):
						diff = (times['new'][1] - times['old'][1])
						dfEvent.at[len(dfEvent), 'duration'] = float(diff)/1000

				# Craving score
				if re.search('VasSlide.VAS', l):
					l_split = l.split(' ')
					dfEvent.at[len(dfEvent), 'rating'] = int(l_split[1])

		# Summary measures
		for t in rowSummary[:4]:
			dfSummary.at[t, 'ratingAvg'] = np.mean(dfEvent.loc[dfEvent['fileName'] == t, 'rating'])
			dfSummary.at[t, 'ratingDuration'] = np.mean(dfEvent.loc[dfEvent['fileName'] == t, 'duration'])
			dfSummary.at[t, 'ratingOmit'] = np.sum(-dfEvent.loc[dfEvent['fileName'] == t, 'ratingOmit'].isnull())
			self.spmStruct['names'].append(t)
			self.spmStruct['onsets'].append(dfEvent.loc[dfEvent['subtype'] == t, 'onsetFromStart'])
			self.spmStruct['durations'].append(dfEvent.loc[dfEvent['subtype'] == t, 'duration'])
			self.spmStruct['names'].append(t + 'Rating')
			self.spmStruct['onsets'].append(dfEvent.loc[dfEvent['fileName'] == t, 'onsetFromStart'])
			self.spmStruct['durations'].append(dfEvent.loc[dfEvent['fileName'] == t, 'duration'])

		dfSummary.at['allAlcohol', 'ratingAvg'] = np.mean(dfEvent.loc[dfEvent['fileName'].isin(allAlcohol), 'rating'])
		dfSummary.at['allAlcohol', 'ratingDuration'] = np.mean(dfEvent.loc[dfEvent['fileName'].isin(allAlcohol), 'duration'])
		dfSummary.at['allAlcohol', 'ratingOmit'] = np.sum(-dfEvent.loc[dfEvent['fileName'].isin(allAlcohol), 'ratingOmit'].isnull())

		# Clean fractions
		dfSummary['ratingAvg'] = [float("{0:.3f}".format(idx)) for idx in dfSummary['ratingAvg']]
		dfSummary['ratingDuration'] = [float("{0:.3f}".format(idx)) for idx in dfSummary['ratingDuration']]

		# Return even and summary data frames
		return {'event': dfEvent, 'summary': dfSummary, 'spm': self.spmStruct}	
						
	def evalRLP(self, fullFileName):
		""" Evaluate RLP behavioral response file """

		# column names
		colEvents = ['block_num', 'trial_num', 'stimulus_type', 'onset', 'duration', 'response_time', 'response', 'correct_response', 'left_image', 'right_image', 'feedback_correct', 'feedback_incorrect', 'feedback_received', 'reinforcement_ratio']
		
		# variables to keep a strings
		keepString = {'^(LeftImage)': 'left_image', '^(RightImage)': 'right_image', '^(FeedbackCorrect)': 'feedback_correct', '^(FeedbackIncorrect)': 'feedback_incorrect'}
		
		# variables to keep as numeric
		keepNumeric = {'Fixation.OnsetTime': 'fix_on', 'Fixation.OffsetTime': 'fix_off', 'Cues.OnsetTime': 'cue_on', 'Cues.OffsetTime': 'cue_off', 'Response.OnsetTime': 'resp_on', 'Response.OffsetTime': 'resp_off', 'Feedback.OnsetTime': 'feed_on', 'Feedback.OffsetTime': 'feed_off', 'Cues.RESP': 'resp', 'Cues.CRESP': 'cresp', 'Cues.RT': 'rt', 'BreakText[0-9]{1}.OnsetTime': 'break_on', 'ScannerWaitPreBlock[0-9]{1}.OnsetTime': 'ready_on'}
		
		
		dfEvent = pd.DataFrame(columns = colEvents) # event dataframe
		trialCounter = 0
		absoluteStart = None # time point (in ms) when paradigm starts
		currTrial = {'trial_num': trialCounter} # current trial information

		# Process fMRI behavioral file
		with io.open(fullFileName,'r', encoding='utf-16') as f:
			for line in f:
				
				l = str(line.rstrip()).lstrip()
				
				# new trial identifier
				if l=='Procedure: Stimulus':
					trialCounter += 1
					if trialCounter > 1:
						dfEvent = self.eventRLP(currTrial, dfEvent, absoluteStart)
					currTrial = {'trial_num': trialCounter}
				
				# pull relevant field for string and numeric values of interest
				strCheck = [v for k,v in keepString.items() if re.search(k,l)]
				numCheck = [v for k,v in keepNumeric.items() if re.search(k,l)]
				
				if strCheck:
					l_split = l.split(': ')
					currTrial[''.join(strCheck)] = l_split[1]
				elif numCheck:
					l_split = l.split(': ')
					if len(l_split) == 2:
						if re.search('Cues.RT',l) and l_split[1] == '0':
							currTrial[''.join(numCheck)] = None # process non-responses
						elif re.search('Fixation.OnsetTime',l) and trialCounter == 1:
							absoluteStart = int(l_split[1]) # first fixation onset is first event of task
							currTrial[''.join(numCheck)] = int(l_split[1]) # update currTrial
						else:
							currTrial[''.join(numCheck)] = int(l_split[1]) # update currTrial
					else:
						currTrial[''.join(numCheck)] = None # update currTrial
				
				if re.search('^(Running)',l):
					l_split = l.split(': ')
					if re.search('Block(.*)List[0-9]{3,4}',l_split[1]):
						currTrial['block_num'] = int(re.search('Block(.*)List',l_split[1]).group(1))
						currTrial['reinforcement_ratio'] = int(re.search('Block[0-9]{1}List(.*)',l_split[1]).group(1))
				
				# onset of built in break periods
				if re.search('BreakText[0-9]{1}.OnsetTime',l):
					dfEvent = self.eventRLP(currTrial, dfEvent, absoluteStart)
					trialCounter += 1
					currTrial['trial_num'] = trialCounter
					currTrial = {k:v for k,v in currTrial.items() if k in ['trial_num','block_num']}
					l_split = l.split(': ')
					currTrial['onset'] = int(l_split[1])
					currTrial['stimulus_type'] = 'break'
				
				# alert to participant that break period is about to end
				if re.search('ScannerWaitPreBlock[0-9]{1}.OnsetTime',l):
					dfEvent = self.eventRLP(currTrial, dfEvent, absoluteStart)
					trialCounter += 1
					currTrial['trial_num'] = trialCounter
					currTrial = {k:v for k,v in currTrial.items() if k in ['trial_num','block_num']}
					l_split = l.split(': ')
					currTrial['onset'] = int(l_split[1])
					currTrial['stimulus_type'] = 'getReady'
		
		# compute durations
		nanIdx = np.where(np.isnan(dfEvent['duration']))
		for elem in nanIdx[0]:
			if elem+1 < len(dfEvent.index):
				dfEvent.at[elem,'duration'] = dfEvent.at[elem+1,'onset']-dfEvent.at[elem,'onset']
			else:
				dfEvent.at[elem,'duration'] = 30
		
		# Return event data frame (summary not written)
		f.close()
		dfEvent.replace(np.nan,'n/a',inplace=True)
		return {'event': dfEvent, 'summary': None}

	def eventRLP(self, currTrial, dfEvent, absoluteStart):
		""" Organize current trial and update dfEvent """
		
		nrep = 4 # rlp trials comprise four events (fixation, cue, response, feedback)
		
		# "break text" and "scanner wait" events handled differently
		if all([i in currTrial.keys() for i in ['trial_num', 'block_num', 'onset', 'stimulus_type']]):
			if 'ready_on' in currTrial.keys():
				currTrial.pop('ready_on')
			
			currTrial['onset'] = (currTrial['onset'] - absoluteStart)/1000
			dfEvent = dfEvent.append(pd.DataFrame(currTrial, index = [0]), ignore_index = True)
		else:	
			
			# compute onsets relative to task start time (absoluteStart)
			currOnsets = [currTrial['fix_on']-absoluteStart, currTrial['cue_on']-absoluteStart, currTrial['resp_on']-absoluteStart, currTrial['feed_on']-absoluteStart]
			
			# convert to seconds
			currOnsets = [i/1000 for i in currOnsets]
			
			# compute durations
			currDurations = [(currTrial['fix_off']-currTrial['fix_on'])/1000, (currTrial['cue_off']-currTrial['cue_on'])/1000, (currTrial['resp_off']-currTrial['resp_on'])/1000, None]
			# convert rt to seconds
			if currTrial['rt'] is not None:
				currTrial['rt'] = currTrial['rt']/1000
			
			# determine feedback received by participant
			if currTrial['resp'] is None:
				feedback_received = 'Blank' # non-response receives "Blank" feedback
			else:
				feedback_received = [currTrial['feedback_incorrect'], currTrial['feedback_correct']][int(currTrial['resp'] == currTrial['cresp'])]
			
			# update currEvent
			currEvent = {'trial_num': np.repeat(currTrial['trial_num'],nrep), 'block_num': np.repeat(currTrial['block_num'],nrep), 'stimulus_type': ['fixation', 'cue', 'response', 'feedback'], 'onset': currOnsets, 'duration': currDurations, 'response_time': ['n/a', currTrial['rt'], 'n/a', 'n/a'], 'response': ['n/a', currTrial['resp'], 'n/a', 'n/a'], 'correct_response': ['n/a', currTrial['cresp'], 'n/a', 'n/a'], 'left_image': ['n/a', currTrial['left_image'], 'n/a', 'n/a'], 'right_image': ['n/a', currTrial['right_image'], 'n/a', 'n/a'], 'feedback_correct': ['n/a', 'n/a', 'n/a', currTrial['feedback_correct']], 'feedback_incorrect': ['n/a', 'n/a', 'n/a', currTrial['feedback_incorrect']], 'feedback_received': ['n/a', 'n/a', 'n/a', feedback_received], 'reinforcement_ratio': ['n/a', 'n/a', 'n/a', currTrial['reinforcement_ratio']]}
			
			# add currEvent df to dfEvent
			dfEvent = dfEvent.append(pd.DataFrame.from_dict(currEvent), ignore_index = True)
		
		# return updated dfEvent
		return(dfEvent)
	
	def idTrim(self, id):
		""" Trim initials from a scan ID name """

		# Prisma
		if re.search('^(p[0-9][0-9][0-9])', id):
			return id[:4]
		# Verio/mMR/mr001
		elif re.search('^([mvn][0-9][0-9][0-9][0-9])', id):
			return id[:5]
		else:
			return id

	def writeRespCsv(self, respOutput):
		""" Write response files to .csv """

		# Time stamp of when .txt file processed (not used, currently)
		self.currTime = time.strftime('%c')

		scannerID_initialsOmit = self.idTrim(self.currSubjID)
		if self.outFileSuffix is not None:
			suffix = '_' + self.outFileSuffix
		else:
			suffix = ''
		eventFileName = '%s_%s_EventResp%s.tsv' % (scannerID_initialsOmit, self.outputName[self.currTaskType], suffix)
		summaryFileName = '%s_%s_SummaryResp.txt' % (scannerID_initialsOmit, self.outputName[self.currTaskType])

		# Write output files
		if respOutput['event'] is not None:
			respOutput['event'].to_csv('/'.join([os.getcwd(), eventFileName]), index = False, sep = '\t')
			print('Event file written: %s ' % ('/'.join([os.getcwd(), eventFileName])))
		if respOutput['summary'] is not None:
			respOutput['summary'].to_csv('/'.join([os.getcwd(), summaryFileName]))
			print('Summary file written: %s ' % ('/'.join([os.getcwd(), summaryFileName])))

	def writeRespMat(self, respOutput):
		""" Write .mat file that can be read into SPM single-subject design matrix """

		scannerID_initialsOmit = self.idTrim(self.currSubjID)
		matFileName = '%s_%sRespSPM.mat' % (scannerID_initialsOmit, self.outputName[self.currTaskType])

		# Currently formatted for alcohol task only
		if self.currTaskType != 'VAS':
			print('Skipping .mat write for %s, %s ' % (scannerID_initialsOmit, self.outputName[self.currTaskType]))
			return

		# Work around
		## I think respData['spm']['names'] is a list and therefore creating these variables should not be necessary but it does not work otherwise (reason unknown).
		onsets = [respData['spm']['onsets'][idx].tolist() for idx in range(len(respData['spm']['onsets']))]
		durations = [respData['spm']['durations'][idx].tolist() for idx in range(len(respData['spm']['durations']))]
		names = np.asarray(respData['spm']['names'], dtype = 'O')
		onsets = np.asarray(onsets, dtype = 'O')
		durations = np.asarray(durations, dtype = 'O')

		# Write output file
		sio.savemat('/'.join([os.getcwd(), matFileName]), {'names': names, 'durations': durations, 'onsets': onsets})
		print('SPM response file written: %s ' % ('/'.join([os.getcwd(), matFileName])))

if __name__ == '__main__':
	""" Script called from linux command line """

a = Response()
inputInfo = a.mapInput(a.input1)
fileNames = a.inputDict[inputInfo[0]](inputInfo[1])
for f in fileNames:
	taskType = a.taskIdentify(f)
	respData = taskType(f)
	a.writeRespCsv(respData)
	a.writeRespMat(respData)
