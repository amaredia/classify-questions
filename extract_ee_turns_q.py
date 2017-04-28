import csv
import re
import sys
import numpy as np
# ee_id from spreadsheet, question number, text of question, answer, TF, start and end time

def parse_filename(filename):
	er = None
	ee = None
	parse_filename = filename.split('-')
	interview_pair = parse_filename[0].replace('p', ' ')
	interview_pair = interview_pair.split()
	turn = parse_filename[1].split('_')[0]
	if turn == 'part1':
		er = interview_pair[0]
		ee = interview_pair[1]
	else:
		er = interview_pair[1]
		ee = interview_pair[0]
	return (er, ee)


#segments question labeled ER to get start time, end time, question #, question, and file name
	#dict (interviewer, interviee), start time, end time, question number, question
def parse_qlabel_file(file):
	pairs = {}
	info_reader = csv.DictReader(open(file, 'rb'), delimiter = '\t', fieldnames = ['', 'filename', 'starttime', 'endtime','text', 'overlap', 'laughter', 'q_num'])
	next(info_reader)
	for row in info_reader:
		interview_pair = parse_filename(row['filename'])
		if interview_pair in pairs:
			pairs[interview_pair] = np.vstack((pairs[interview_pair], [row['starttime'], row['endtime'], row['q_num'], row['text']]))
		else:
			pairs[interview_pair] = np.asarray([row['starttime'], row['endtime'], row['q_num'], row['text']])

	return pairs


#turns questions from question file into dictionary
def read_qfile(file):
	questions = {}
	q_num = 1
	q_reader = csv.reader(open(file, 'rb'))
	next(q_reader)
	for q in q_reader:
		questions[q_num] = q
		q_num+=1
	return q_num


#gets pid, start time, end time, interviewee text for each question
def parse_ee_file(file):
	pairs = {}
	info_reader = csv.DictReader(open(file, 'rb'), delimiter='\t', fieldnames = ['filename', 'tiername', 'starttime', 'endtime','text'])
	next(info_reader)
	for row in info_reader:

		interview_pair = parse_filename(row['filename'])
		if interview_pair in pairs:
			pairs[interview_pair] = np.vstack((pairs[interview_pair], [row['starttime'], row['endtime'], row['text']]))
		else:
			pairs[interview_pair] = np.asarray([row['starttime'], row['endtime'], row['text']])

	return pairs


#read conf file
def parse_conf_file(file):
	pairs = {}
	info_reader = csv.DictReader(open(file, 'rb'), fieldnames = ['', 'er', 'ee', 'q_num','TF', 'guess', 'confidence'])
	next(info_reader)
	for row in info_reader:
		pairs[(row['er'], row['ee'], row['q_num'])] = {'TF': row['TF'], 'guess': row['guess'], 'confidence': row['confidence']}
	return pairs


#combine the question labeled with the EE turns
#dict(interviewer, interviewee, q_num) start time, end time, answer, question, answer
def combine_er_ee(er_files, ee_files):
	filler_words = ['oh', 'um', 'mm', 'mmm', 'hmm', 'hm', 'ah', 'uh',  'um', 'er', 'eh', 'ha']
	combined = {}
	for pair in ee_files:
		er_turns = er_files[pair]
		ee_turns = ee_files[pair]
		er_turns.sort()
		ee_turns.sort()
		
		er_pos = 0
		ee_pos = 0
		# er_start = er_turns[er_pos][0]
		# ee_start = ee_turns[ee_pos][0]
		
		while er_pos < len(er_turns) and ee_pos < len(ee_turns):
			er_start = er_turns[er_pos][0]
			ee_start = ee_turns[ee_pos][0]
			ee_text = ee_turns[ee_pos][2]
			if float(er_start) > float(ee_start) or ee_text.lower() in filler_words:
				ee_pos+=1
				#ee_start = ee_turns[ee_pos][0]
			else:
				#add to dict (er, ee, q_num) = [ee_start, ee_end, ee_text]
				ee_end = ee_turns[ee_pos][1]
				er_qtext = er_turns[er_pos][3]

				combined[(pair[0], pair[1], er_turns[er_pos][2])] =  {'ee_start': ee_start, 'ee_end': ee_end, 'er_qtext': er_qtext, 'ee_text': ee_text}
				er_pos +=1
				ee_pos +=1
				#er_start = er_turns[er_pos][0]

	return combined


#combine conf and EE
def combine_conf_turns(conf, turns):
	for pair in conf:
		if pair in turns:
			conf[pair] = dict(conf[pair], **turns[pair])
			conf[pair]['q_num'] = pair[2]
	return conf


#write to csv
def write_to_file(conf, output, er_ee_combined):
	with open(output, 'wb') as csvfile:
		content_writer = csv.writer(open(output, 'wb'))
		content_writer.writerow(
			['interviewer', 'interviewee', 'question', 'starttime', 'endtime', 'question_text', 'text', 'TF', 'guess'])
		for key in sorted(conf.iterkeys()):
			# ee_id from spreadsheet, question number, text of question, answer, TF, start and end time
			data = conf[key]
			if key in er_ee_combined:
				out = [key[0], key[1], key[2], data['ee_start'], data['ee_end'], data['er_qtext'], data['ee_text'], data['TF'], data['guess']]
			else:
				out = [key[0], key[1], key[2], ' ', ' ', ' ', ' ', data['TF'], data['guess']]

			content_writer.writerow(out)


def convert_list(d):
	for key in d:
		d[key] = d[key].tolist()
	return d


def main():
	if len(sys.argv) != 5:
		print "args must be: <interviewer question labelled file> <interviewee turns file> <conf file> <output file>"
		exit()
	er_qfile = sys.argv[1]
	ee_turn_file = sys.argv[2]
	conf_file = sys.argv[3]
	output = sys.argv[4]

	er_cleaned = parse_qlabel_file(er_qfile)
	ee_cleaned = parse_ee_file(ee_turn_file)
	
	er_cleaned = convert_list(er_cleaned)
	ee_cleaned = convert_list(ee_cleaned)

	conf_cleaned = parse_conf_file(conf_file)

	er_ee_combined = combine_er_ee(er_cleaned, ee_cleaned)
	conf_combined = combine_conf_turns(conf_cleaned, er_ee_combined)

	write_to_file(conf_combined, output, er_ee_combined)


if __name__ == "__main__":
    main()

