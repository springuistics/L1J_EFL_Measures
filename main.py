import os
import sys
import spacy
import re
from collections import Counter
from pathlib import Path
import logging
import csv

logger = logging.getLogger('L1J_EFL_Measures')

def safe_division(x, y):
    """
    Safely divide numbers; i.e., give a score of 0 if numerator or denominator is 0
    :param x:
    :param y:
    :return:
    """
    if float(x) == 0 or float(y) == 0:
        return 0
    return float(x) / float(y)


def process_lex(words, lemmas, pos, stop, wordranks):
    """
    Provides counts of lexical complexity that are predictive of L1 Japanese learners' proficiency, as per (forthcoming).
    Modeled on Lu (2012) and Spring and Johnson (2022)
    :param spacy_words:
    :param spacy_lemmas:
    :param spacy_pos:
    :param spacy_stop:
    :param wordranks:
    :return:
    """
    lemmalist = []
    wordtokens = 0
    wordtypes = 0
    swordtypes = 0
    swordtokens = 0
    verbtypes = 0
    verbtokens = 0
    sverbtypes = 0
    sverbtokens = 0
    lextokens = 0
    lextypes = 0
    slextokens = 0
    slextypes = 0

    for i in range(0, len(words)):
        pos_item = pos[i]
        lemma = lemmas[i]

        if pos_item not in ["PUNCT", "SYM", "X", "SPACE", ".", ",", "!", "?", ":", ";", "-", " ", "¥n"]:
            if lemma not in lemmalist:
                wordtypes += 1
                wordtokens += 1
                if pos_item[0] in ["N", "J", "R", "V"]:
                    lextypes += 1
                    lextokens += 1
                    if lemma not in wordranks[:2000]:
                        slextypes += 1
                        slextokens += 1
                if pos_item[0] == "V":
                    verbtypes += 1
                    verbtokens += 1
                if lemma not in wordranks[:2000]:
                    swordtypes += 1
                    swordtokens += 1
                    if pos_item[0] == "V":
                        sverbtypes += 1
                        sverbtokens += 1

                lemmalist.append(lemma)

            else:
                wordtokens += 1
                if pos_item[0] in ["N", "J", "R", "V"]:
                    lextokens += 1
                    if lemma not in wordranks[:2000]:
                        slextokens += 1
                if pos_item[0] == "V":
                    verbtokens += 1
                if lemma not in wordranks[:2000]:
                    swordtokens += 1
                    if pos_item[0] == "V":
                        sverbtokens += 1


    if words == []:
        LS = 'error - file could not be read!'
    else:
        LS = safe_division(slextokens, lextokens)

    VS = safe_division((sverbtypes ** 2), verbtokens)
    CVS = safe_division(sverbtypes, ((2 * verbtokens) ** 0.5))
    NDW = wordtypes
    CTTR = safe_division(wordtypes, ((2 * wordtokens) ** 0.5))
    SVV = safe_division((verbtypes ** 2), verbtokens)
    CVV = safe_division(verbtypes, ((2 * verbtokens) ** 0.5))

    return {'LS': LS, 'VS': VS, 'CVS': CVS, 'NDW': NDW, 'CTTR': CTTR, 'SVV': SVV, 'CVV': CVV}

def process_sdm(spacy_words, spacy_deps, word_count):
    """
    Counts number of supporting detail markers (SDMs - see Spring 2023) by creating lists of words and n2~4 grams, and
    comparing these against SDM n~n4 grams. Also provides C2SDM_S and C2SDM_C as per Spring 2023. Uses deps to do this.
    :param spacy_words:
    :param spacy_deps:
    :param word_count:
    :return:
    """

    count = 0
    n2 = []
    n3 = []
    n4 = []
    ac1 = ["including", "include", "includes", "contrary", "illustrate", "illustrates", "exemplifies", "exemplify",
           "meaning", "misinterpret", "misinterprets", "misinterpreting", "because", "since", "therefore", "cause",
           "causes", "yield", "yields", "moreover", "futhermore", "however", "although", "nevertheless", "yet",
           "though", "either", "instead", "if", "without", "specifically", "additionally", "consequently"]
    ac2 = ["in addition", "caused by", "for one", "in addition", "for instance", "for example", "regarded as",
           "seeing that", "leads to", "lead to", "leading to", "divided into", "fall into", "falls into",
           "falling into", "considered as", "this implies", "this suggests", "brings about", "bring about",
           "brought about", "bringing about", "due to", "based on", "so that", "such as", "into account", "points to",
           "point to", "pointing to", "pointed to", "points out", "point out", "pointing out", "pointed out",
           "refers to", "refer to", "regarded as"]
    ac3 = ["wide range of", "with respect to", "to distinguish between", "as explained by", "this means that",
           "it follows that", "also known as", "the difference between", "on account of", "in order to",
           "the reason for", "the reason why", "in this respect", "in spite of", "so as to", "factors of this",
           "a number of", "matter of fact", "in other words", "in respect to"]
    ac4 = ["at the same time", "it turns out that", "for the purpose of", "on the other hand", "it should be noted",
           "in the case that", "can be seen by", "at the same time"]

    for word in range((len(spacy_words) - 1)):
        temp = [spacy_words[j] for j in range(word, word + 2)]
        temp = ' '.join(temp)
        temp = temp.lower()
        n2.append(temp)

    for word in range((len(spacy_words) - 2)):
        temp = [spacy_words[j] for j in range(word, word + 3)]
        temp = ' '.join(temp)
        temp = temp.lower()
        n3.append(temp)

    for word in range((len(spacy_words) - 3)):
        temp = [spacy_words[j] for j in range(word, word + 4)]
        temp = ' '.join(temp)
        temp = temp.lower()
        n4.append(temp)

    for entry in spacy_words:
        entry2 = entry.lower()
        if entry2 in ac1:
            count += 1

    for entry in n2:
        if entry in ac2:
            count += 1

    for entry in n3:
        if entry in ac3:
            count += 1

    for entry in n4:
        if entry in ac4:
            count += 1

    count_dict = dict(Counter(spacy_deps))

    try:
        s = count_dict['ROOT']
    except KeyError:
        s = 0

    try:
        nsubj = count_dict['nsubj']
    except KeyError:
        nsubj = 0

    try:
        nsubjpass = count_dict['nsubjpass']
    except KeyError:
        nsubjpass = 0

    try:
        csubj = count_dict['csubj']
    except KeyError:
        csubj = 0

    try:
        csubjpass = count_dict['csubjpass']
    except KeyError:
        csubjpass = 0

    c = nsubj + nsubjpass + csubj + csubjpass

    SDM = count
    C2SDM_S = count - (safe_division(count, s))
    C2SDM_C = count - (safe_division(count, c))

    return {'SDM': SDM, 'C2SDM_S': C2SDM_S, 'C2SDM_C': C2SDM_C}


def process_syn(tag, dep, w):
    count_dict = dict(Counter(dep))

    try:
        s = count_dict['ROOT']
    except KeyError:
        s = 0

    try:
        nsubj = count_dict['nsubj']
    except KeyError:
        nsubj = 0

    try:
        nsubjpass = count_dict['nsubjpass']
    except KeyError:
        nsubjpass = 0

    try:
        csubj = count_dict['csubj']
    except KeyError:
        csubj = 0

    try:
        csubjpass = count_dict['csubjpass']
    except KeyError:
        csubjpass = 0

    try:
        ccomp = count_dict['ccomp']
    except KeyError:
        ccomp = 0

    try:
        xcomp = count_dict['xcomp']
    except KeyError:
        xcomp = 0

    try:
        adverbclause = count_dict['advcl']
    except KeyError:
        adverbclause = 0

    try:
        acl = count_dict['acl']
    except KeyError:
        acl = 0

    try:
        relcl = count_dict['relcl']
    except KeyError:
        relcl = 0

    nmod_keys = ['nmod', 'npmod', 'tmod', 'poss']
    nmod = 0
    for k in nmod_keys:
        if k in count_dict:
            nmod += count_dict[k]

    omod_keys = ['advmod', 'amod', 'appos']
    omod = 0
    for k in omod_keys:
        if k in count_dict:
            omod += count_dict[k]

    cord_keys = ['conj', 'cc']
    cord = 0
    for k in cord_keys:
        if k in count_dict:
            cord += count_dict[k]

    dc_keys = ['acl', 'relcl', 'advcl', 'ccomp']
    dc = 0
    for k in dc_keys:
        if k in count_dict:
            dc += count_dict[k]

    # compute the additional syntactic complexity indices
    clause = nsubj + nsubjpass + csubj + csubjpass
    T = clause - (adverbclause + relcl + acl)
    VP = ccomp + clause
    passives = nsubjpass + csubjpass
    allmod = nmod + omod
    CSTR = (dc + xcomp + cord + nmod + omod)
    adjcl = acl + relcl
    mls = safe_division(w, s)
    mlt = safe_division(w, T)
    mlc = safe_division(w, clause)
    c_s = safe_division(clause, s)
    vp_t = safe_division(VP, T)
    c_t = safe_division(clause, T)
    t_s = safe_division(T, s)
    co_s = safe_division(cord, s)
    co_t = safe_division(cord, T)
    co_c = safe_division(cord, clause)
    adv_s = safe_division(adverbclause, s)
    adv_t = safe_division(adverbclause, T)
    adv_c = safe_division(adverbclause, clause)
    adj_s = safe_division(adjcl, s)
    adj_t = safe_division(adjcl, T)
    adj_c = safe_division(adjcl, clause)
    dc_s = safe_division(dc, s)
    dc_t = safe_division(dc, T)
    dc_c = safe_division(dc, clause)
    pass_s = safe_division(passives, s)
    pass_t = safe_division(passives, T)
    pass_c = safe_division(passives, clause)
    allmod_s = safe_division(allmod, s)
    allmod_t = safe_division(allmod, T)
    allmod_c = safe_division(allmod, clause)
    CSTR_s = safe_division(CSTR, s)
    CSTR_t = safe_division(CSTR, T)
    CSTR_c = safe_division(CSTR, clause)

    return {'W': w, 'MLS': mls, 'MLT': mlt, 'C_T': c_t, 'VP_T': vp_t, 'CO_T': co_t, 'DC_T': dc_t, 'CSTR_S': CSTR_s, 'CSTR_T': CSTR_t}

def process_phrase(spacy_deps, spacy_pos, spacy_heads, spacy_words, spacy_children, spacy_lemmas):
    """
    Creates phrasal complexity measures based on Kyle 2016 and Kyle & Crossley 2018.
    Though Kyle & Crossley (etc) offer a lot of measures, this program just focuses on ones that are indicative of
    L2 proficiency, as per (forthcoming paper)
    :param spacy_tags:
    :param spacy_deps:
    :return:
    """

    def stdev(data):
        """
        Calculates standard deviation of a data set. Would rather write it than wazawaza import just for this
        :param data:
        :return:
        """
        n = len(data)
        if n == 0 or n == 1:
            return 0

        else:
            mean = sum(data) / n
            dev = [(x - mean) ** 2 for x in data]
            variance = sum(dev) / n
            mystdev = variance ** 0.5
            return mystdev

    def safe_avg(data):
        n = len(data)
        if n == 0:
            return 0

        else:
            mean = sum(data) / n
            return mean

    nom_deps = 0
    NN_deps = []
    NN = 0
    pobj = 0
    pobjs = []
    pobj_deps = []
    pobj_prep_deps = []
    preps = []
    prep_pobj = 0
    stative_verbs = ["be", "exist", "appear", "feel", "hear", "look", "see", "seem", "belong", "have", "own", "possess", "like", "want", "wish", "prefer", "love", "hate"]
    satellites =["across", "after", "along", "apart", "around", "about", "away", "back", "down", "under", "in", "into", "off", "on", "onto", "out", "over", "above", "through", "to", "together", "below", "up"]
    satellite_framings = 0
    verbs = 0

    for i in range(0, len(spacy_deps)):
        dep = spacy_deps[i]
        pos = spacy_pos[i]
        word = spacy_words[i]
        head = spacy_heads[i]
        head = f'{head}'

        if pos == "VERB":
            verbs += 1

        if dep in ["nsubj", "nsubjpass", "agent", "ncomp", "dobj", "iobj", "pobj"]:
            NN += 1
            counter = 0
            tempkids = []
            for x in spacy_children[i]:
                tempkids.append(f'{x}')
                for y in range(0, len(spacy_deps)):
                    match_W = spacy_words[y]
                    find_dep = spacy_deps[y]
                    if match_W in tempkids:
                        if find_dep in ["det", "amod", "prep", "poss", "vmod", "nn", "rcmod", "advmod", "conj_and", "conj_or"]:
                            counter += 1
            NN_deps.append(counter)


        if dep in ["det", "amod", "prep", "poss", "vmod", "nn", "rcmod", "advmod", "conj_and", "conj_or"]:
            nom_deps += 1

        if dep == "pobj":
            pobj += 1
            tempdesp = []
            counter = 0
            counter2 = 0
            for x in spacy_children[i]:
                tempdesp.append(f'{x}')
                for y in range(0, len(spacy_deps)):
                    john = spacy_words[y]
                    sally = spacy_deps[y]
                    if john in tempdesp:
                        if sally in ["det", "amod", "prep", "poss", "vmod", "nn", "rcmod", "advmod", "conj_and", "conj_or"]:
                            counter += 1
                        if sally == "prep":
                            prep_pobj += 1
                            counter2 += 1
            pobjs.append({"word": word, "index": i})
            pobj_deps.append(counter)
            pobj_prep_deps.append(counter2)


        if dep == "prep":
            if word in satellites:
                for y in range(0, i):
                    word_pos = spacy_pos[y]
                    word_match = spacy_words[y]
                    word_lemma = spacy_lemmas[y]
                    if word_match == head and word_pos == "VERB" and (i-y) < 10:
                        if word_lemma not in stative_verbs:
                            satellite_framings += 1

            preps.append({"word": word, "index": i})

        if pos == "ADJ":
            for y in range(0, i):
                word_pos = spacy_pos[y]
                word_match = spacy_words[y]
                word_lemma = spacy_lemmas[y]
                if word_match == head and word_pos == "VERB" and (i-y) < 10:
                    if word_lemma not in stative_verbs:
                        satellite_framings += 1

    for items in preps:
        id = items["index"]
        for i in range(0, len(spacy_deps)):
            if id == i:
                if spacy_deps[i-1] == "pobj":
                    prep_pobj += 1


    av_nom_deps_NN = safe_division(nom_deps, NN)
    av_nom_deps_2 = safe_avg(NN_deps)
    Prep_pobj_deps_NN = safe_division(prep_pobj, pobj)
    Prep_pobj_deps_2 = safe_avg(pobj_prep_deps)
    Av_pobj_deps = safe_avg(pobj_deps)
    Pobj_NN_SD = stdev(pobj_deps)
    AnotherSF = satellite_framings + prep_pobj



    return {'PC1': av_nom_deps_NN, 'PC1.5': av_nom_deps_2, 'PC2': Prep_pobj_deps_NN, 'PC2.5': Prep_pobj_deps_2, 'PC3': Pobj_NN_SD, 'SFraming': satellite_framings, 'SF2': AnotherSF}




def get_scores(filename, spacy_output, wordranks):
    """
    Splits SpaCy into relevant elements and utilizes SDM counts and wordranks to create scores.
    Scores come from a culmination of Lu 2010, 2012, Spring 2023, and Kyle and Crossley 2018
    Only scores that are relevant for a large majority of L1 Japanese EFL learners are selected from the above
    (see forthcoming paper)
    :param spacy_output:
    :param wordranks:
    :return:
    """
    spacy_words = []
    spacy_lemmas = []
    spacy_pos = []
    spacy_tags = []
    spacy_deps = []
    spacy_stop = []
    spacy_heads = []
    spacy_children = []

    for idx, token in enumerate(spacy_output):
        spacy_words.append(spacy_output[idx].text)
        spacy_lemmas.append(spacy_output[idx].lemma_)
        spacy_pos.append(spacy_output[idx].pos_)
        spacy_tags.append(spacy_output[idx].tag_)
        spacy_deps.append(spacy_output[idx].dep_)
        spacy_stop.append(spacy_output[idx].is_stop)
        spacy_heads.append(spacy_output[idx].head)
        kids = []
        for x in spacy_output[idx].children:
            kids.append(x)
        spacy_children.append(kids)

    word_count = len([token for token in spacy_output if
                      token.is_alpha or token.shape_ == 'dd'])  # dd is spacy's definition for digits.

    lexical_results = process_lex(spacy_words, spacy_lemmas, spacy_tags, spacy_stop, wordranks)
    SDM_results = process_sdm(spacy_words, spacy_deps, word_count)
    phrasal_results = process_phrase(spacy_deps, spacy_pos, spacy_heads, spacy_words, spacy_children, spacy_lemmas)
    trad_synt_results = process_syn(spacy_tags, spacy_deps, word_count)
    filename_res = {'filename': filename}

    return {'filename': filename_res, 'lex_res': lexical_results, 'SDM_res': SDM_results, 'phr_res': phrasal_results, 'syn_res': trad_synt_results}


def write_header_and_data_to_file(header, data, output_filename):
    """

    :type header - a string.
    :param header:
    :type data - list of strings.
    :param data:
    :param output_filename:
    :return:
    """
    assert Path(output_filename).parent.is_dir(), f'The directory: {Path(output_filename).parent.absolute()} does ' \
                                                  f'not exist. Cannot create output file. Please create the above ' \
                                                  f'directory, or choose another file location.'
    with open(output_filename, 'w', encoding='utf8', newline='') as output_file:
        output_file.write(header)
        for d in data:
            output_file.write(d)
    logger.info(f'{len(data)} lines of output written to: {output_filename}.')


def read_coca_frequent_data(i_filename='coca_frequent_words.csv'):
    """
    rank	lemma	PoS
    1	the	a
    2	be	v
    3	and	c

    :param i_filename:
    :type i_filename:
    :return:
    :rtype:
    """
    data = []
    with open(i_filename, 'r', newline='\n', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        first = True
        for row in reader:
            if first:
                first = False
                continue
            data.append(row[1])
    return data


def read_input_text(filename):
    """
    Takes file name with encoding as utf8
    If the file encounters non-utf8 characters, it returns an error
    :param filename:
    :return:
    """
    text_lines = ""
    try:
        with open(filename, 'r', encoding='utf8') as f:
            text_lines = f.readlines()
            logger.info(f'Read file: {filename} - {len(text_lines)} lines read.')
    except:
        text_lines = 'unrecognized characters in file - unable to process'
        logger.warning(f'Filename:{filename} could not be read. Returning empty read-lines.')

    finally:
        return text_lines


def process(file_path: str, filename, wordranks):
    """
    Simply preps files by sending files to get read and handling encoding errors.
    Also gets singular analysis from SpaCy, then sends to a score getter.
    :param file_path:
    :param ac: just taken from SDM list (see Spring 2023)
    :param wordranks: read once from common words list (see Spring & Johnson 2022)
    :return:
    """
    text_lines = read_input_text(file_path)
    input_text = ''.join(text_lines)

    if input_text == 'unrecognized characters in file - unable to process':
        results = get_scores(filename, "", wordranks)
        return results
    else:
        nlp = spacy.load("en_core_web_lg")
        analysis = nlp(input_text)
        results = get_scores(filename, analysis, wordranks)
        return results


def write_header_and_data_to_file(header, data, output_filename):
    """

    :type header - a string.
    :param header:
    :type data - list of strings.
    :param data:
    :param output_filename:
    :return:
    """
    assert Path(output_filename).parent.is_dir(), f'The directory: {Path(output_filename).parent.absolute()} does ' \
                                                  f'not exist. Cannot create output file. Please create the above ' \
                                                  f'directory, or choose another file location.'
    with open(output_filename, 'w', encoding='utf8', newline='') as output_file:
        output_file.write(header)
        for d in data:
            output_file.write(d)
    logger.info(f'{len(data)} lines of output written to: {output_filename}.')

def build_header(scores):
    header = ''
    for key in scores[0]:
        for k in scores[0][key].keys():
            header += f'{k},'
    header += '\n'
    return header


def stringify_scores(scores):
    """
    Scores array is a list of dictionaries
    Format:
    [{'filename': filename, 'scores': spacy_scores, 'syntax': spacy_syntax_results},...]

    :param scores:
    :return:
    """
    string_scores = ''
    for sc in scores:
        for key in sc.keys():
            for v in sc[key].values():
                string_scores += f'{v},'
        string_scores += '\n'
    return string_scores


def list_stringify_scores(scores):
    string_scores_list = []

    for sc in scores:
        string_score = ''
        for key in sc.keys():
            for v in sc[key].values():
                string_score += f'{v},'
        string_score += '\n'
        string_scores_list.append(string_score)
    return string_scores_list


def main(input_path):
    input_filepath = os.path.join(os.getcwd(), input_path)
    wordranks = read_coca_frequent_data()

    scores = []
    for fdx, filename in enumerate(os.listdir(input_filepath)):
        if filename.endswith('.txt'):
            result = process(os.path.join(input_filepath, filename), filename, wordranks)
            scores.append(result)


    header = build_header(scores)
    string_scores = list_stringify_scores(scores)
    write_header_and_data_to_file(header, string_scores, os.path.join(os.getcwd(),
                                                                          f'./output/L1J_EFL_Scores_{len(scores)}.csv'))


if __name__ == '__main__':
    main(sys.argv[1])
