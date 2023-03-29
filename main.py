import os
import sys
import spacy
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
    Provides counts of lexical complexity that are predictive of L1 Japanese learners' proficiency, as per (forthcoming)
    Modeled on Lu (2012) and Spring and Johnson (2022)
    :param words: list of words from SpaCy tokens
    :param lemmas: list of lemma from SpaCy tokens
    :param pos: list of POS tags from SpaCy tokens
    :param stop: List of boolean values for whether the word is a 'stop' word as per SpaCy tagging
    :param wordranks: From COCA corpus
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

    if not words:
        LS = 'error - file could not be read!'
    else:
        LS = safe_division(slextokens, lextokens)

    w_count = len(lemmas)
    msttr = 0
    if w_count < 12:
        msttr = safe_division(wordtypes, wordtokens)
    else:
        z = 11
        dumbcount = 0
        while z <= w_count:
            dumbcount += 1
            templist = []
            tempwordtypes = 0
            tempwordtokens = 0
            for i in range(z-11, z):
                templemma = lemmas[i]
                temppos_item = pos[i]
                if temppos_item not in ["PUNCT", "SYM", "X", "SPACE", ".", ",", "!", "?", ":", ";", "-", " ", "¥n"]:
                    if templemma not in templist:
                        tempwordtypes += 1
                        tempwordtokens += 1
                        templist.append(templemma)

                    else:
                        tempwordtokens += 1
            tempmsttr = safe_division(tempwordtypes, tempwordtokens)
            msttr = ((msttr * (dumbcount-1)) + tempmsttr) / dumbcount
            z += 1

    VS = safe_division((sverbtypes ** 2), verbtokens)
    CVS = safe_division(sverbtypes, ((2 * verbtokens) ** 0.5))
    NDW = wordtypes
    CTTR = safe_division(wordtypes, ((2 * wordtokens) ** 0.5))
    SVV = safe_division((verbtypes ** 2), verbtokens)
    CVV = safe_division(verbtypes, ((2 * verbtokens) ** 0.5))

    return {'LS': LS, 'VS': VS, 'CVS': CVS, 'NDW': NDW, 'CTTR': CTTR, 'SVV': SVV, 'CVV': CVV, 'MSTTR11': msttr}


def process_sdm(spacy_words, spacy_deps):
    """
    Counts number of supporting detail markers (SDMs - see Spring 2023) by creating lists of words and n2~4 grams, and
    comparing these against SDM n~n4 grams. Also provides C2SDM_S and C2SDM_C as per Spring 2023. Uses deps to do this.
    :param spacy_words: word parsed from SpaCy
    :param spacy_deps: dependencies of said words
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
    ac2 = ["in addition", "caused by", "for one", "for instance", "for example", "regarded as",
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
           "in the case that", "can be seen by"]

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
    T = s + ccomp
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
    Creates phrasal complexity measures based on Kyle (2016) and Kyle & Crossley (2018). Specifically, going after
    Av_nom_deps_NN, Prep_pobj_deps_NN, and Pobj_NN_SD (as of 10/14/2022) because though TAASSC (Kyle, 2016) offer a lot
    of measures, this program just focuses on a few that are indicative of L2 proficiency for L1 Japanese EFL learners,
    and so far these are consistently correlated with both as per (forthcoming paper). The NN means that pronouns and
    proper nouns are discluded from the counts.
    Also, does a rough calculation of Satellite-Framing, which can be considered a type of phrasal complexity.
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
            variance = sum(dev) / (n - 1)
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
    phrase_deps = []
    phrases = 0
    pobj = 0
    pobj_deps = []
    pobj_prep_deps = []
    preps = []
    prep_pobj = 0
    prep_pobj2 = 0
    stative_verbs = ["be", "exist", "appear", "feel", "hear", "look", "see", "seem", "belong", "have", "own", "possess",
                     "like", "live", "want", "wish", "prefer", "love", "hate", "make", "become", "meet", "depend",
                     "fit", "touch", "matter", "lay", "lie", "find"]
    satellites = ["aboard", "above", "across", "after", "against", "ahead", "along", "amid", "among",
                  "amongst", "around", "aside", "away", "back", "before", "behind", "below", "beneath", "beside",
                  "between", "beyond", "down", "in", "inside", "into", "near", "off", "on", "onto", "opposite",
                  "out", "outside", "over", "past", "through", "toward", "towards", "together", "under",
                  "underneath", "up", "upon"]
    likely_dates = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
                    "November", "December", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday",
                    "Sunday"]
    satellite_framings = 0
    verbs = 0
    list_o_verbs = []
    list_o_vb_lemmas = []

    for i in range(0, len(spacy_pos)):
        pos = spacy_pos[i]
        verb_lemma = spacy_lemmas[i]
        verb = spacy_words[i]
        if pos == "VERB":
            list_o_verbs.append(verb)
            list_o_vb_lemmas.append(verb_lemma)


    for i in range(0, len(spacy_deps)):
        dep = spacy_deps[i]
        pos = spacy_pos[i]
        word = spacy_words[i]
        head = spacy_heads[i]
        head = f'{head}'

        if pos == "VERB":
            verbs += 1

        if dep in ["nsubj", "nsubjpass", "agent", "ncomp", "dobj", "iobj", "pobj"] and pos == "NOUN":
            phrases += 1
            counter = 0
            tempkids = []
            for x in spacy_children[i]:
                tempkids.append(f'{x}')
            for child in tempkids:
                for x in range(0, len(spacy_deps)):
                    word_match = spacy_words[x]
                    dep_match = spacy_deps[x]
                    if word_match == child:
                        if dep_match in ["det", "amod", "prep", "poss", "vmod", "nn", "rcmod", "advmod", "conj_and", "conj_or"]:
                            counter += 1
                            break
            phrase_deps.append(counter)

        if dep == "pobj" and pos == "NOUN":
            pobj += 1
            tempdesp = []
            counter = 0
            counter2 = 0
            for x in spacy_children[i]:
                tempdesp.append(f'{x}')
            for child in tempdesp:
                for x in range(0, len(spacy_deps)):
                        word_match = spacy_words[x]
                        dep_match = spacy_deps[x]
                        if word_match == child:
                            if dep_match in ["det", "amod", "prep", "poss", "vmod", "nn", "rcmod", "advmod", "conj_and", "conj_or"]:
                                counter += 1
                            if dep_match == "prep":
                                prep_pobj += 1
                                counter2 += 1
                            break
            pobj_deps.append(counter)
            pobj_prep_deps.append(counter2)

        if pos == "ADJ": #handles adjectives as satellites
            if head in list_o_verbs:
                for y in range(0, len(list_o_verbs)):
                    word_match = list_o_verbs[y]
                    word_lemma = list_o_vb_lemmas[y]
                    if word_match == head and word_lemma not in stative_verbs:
                        x = 0
                        while x < i:
                            double_match = spacy_words[x]
                            if head == double_match:
                                satellite_framings += 1
                                break
                            else:
                                x += 1


        if pos == "ADP" and word in satellites: #handles most satellites
            if head in satellites:
                satellite_framings += 1
            elif head in list_o_verbs:
                if i < (len(spacy_words) - 1):
                    if word in ["in", "into", "on", "onto"]:
                        if spacy_words[i+1] not in likely_dates and spacy_pos[i+1] != "NUM" and spacy_pos[i+1] != "PROPN":
                            for y in range(0, len(list_o_verbs)):
                                word_match = list_o_verbs[y]
                                word_lemma = list_o_vb_lemmas[y]
                                if word_match == head and word_lemma not in stative_verbs:
                                    satellite_framings += 1
                                    break
                else:
                    for y in range(0, len(list_o_verbs)):
                        word_match = list_o_verbs[y]
                        word_lemma = list_o_vb_lemmas[y]
                        if word_match == head and word_lemma not in stative_verbs:
                            satellite_framings += 1
                            break

        if pos == "ADV" and word in satellites: #handles particles marked as adverbs as satellites
            if head in satellites:
                satellite_framings += 1
            elif head in list_o_verbs:
                for y in range(0, len(list_o_verbs)):
                    word_match = list_o_verbs[y]
                    word_lemma = list_o_vb_lemmas[y]
                    if word_match == head and word_lemma not in stative_verbs:
                        satellite_framings += 1
                        break


    av_nom_deps_NN = safe_avg(phrase_deps)
    Prep_pobj_deps_NN = safe_avg(pobj_prep_deps)
    Pobj_NN_SD = stdev(pobj_deps)

    return {'PC1': av_nom_deps_NN, 'PC2': Prep_pobj_deps_NN, 'PC3': Pobj_NN_SD, 'SFraming': satellite_framings}


def get_scores(filename, spacy_output, wordranks):
    """
    Splits SpaCy into relevant elements and utilizes SDM counts and wordranks to create scores.
    Scores come from a culmination of Lu 2010, 2012, Spring 2023, and Kyle and Crossley 2018
    Only scores that are relevant for a large majority of L1 Japanese EFL learners are selected from the above
    (see forthcoming paper)
    :param filename: passes in the file name
    :param spacy_output: the tokenized information provided by SpaCy
    :param wordranks: from COCA corpus
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
    SDM_results = process_sdm(spacy_words, spacy_deps)
    phrasal_results = process_phrase(spacy_deps, spacy_pos, spacy_heads, spacy_words, spacy_children, spacy_lemmas)
    trad_synt_results = process_syn(spacy_tags, spacy_deps, word_count)
    filename_res = {'filename': filename}

    return {'filename': filename_res, 'lex_res': lexical_results, 'SDM_res': SDM_results, 'phr_res': phrasal_results, 'syn_res': trad_synt_results}


def write_header_and_data_to_file(header, data, output_filename):
    """

    :type header - a string.
    :param header: All the names of the variables from various types of analyzers
    :type data - list of strings.
    :param data: The scores of the variables from the various analyzers
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
    :param file_path: path of file
    :param filename: actual file name
    :param wordranks: read once from common words list taken from COCA corpus (see Spring & Johnson 2022)
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
    Based on original function in Lu (2010, 2012), later modified in Spring & Johnson (2022)
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
