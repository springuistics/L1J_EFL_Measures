import os
import sys
import spacy
import re
from collections import Counter
from pathlib import Path

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

def read_input_text(filename):
    """
    Takes file name with encoding as utf8
    :param filename:
    :return:
    """
    with open(filename, 'r', encoding='utf8') as f:
        text_lines = f.readlines()
    logger.info(f'Read file: {filename} - {len(text_lines)} lines read.')
    return text_lines

def check_mode(input_filepath):
    """
    Determines if program should run file mode or batch mode.
    :param input_filepath:
    :return:
    """
    assert os.path.exists(input_filepath), f'{input_filepath} does not exist.'
    if os.path.isdir(input_filepath):
        mode = 'directory'
    elif os.path.isfile(input_filepath):
        mode = 'file'
    else:
        assert False, f'{input_filepath} is not a file nor a directory.'
    logger.info(f'{mode} mode recognized. Loading data from: {input_filepath}')
    return mode

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

def prepare_empty_results():
    return {
        'wordtypes': {}, 'wordtokens': 0,
        'swordtypes': {}, 'swordtokens': 0,

        'lextypes': {}, 'lextokens': 0,
        'slextypes': {}, 'slextokens': 0,

        'verbtypes': {}, 'verbtokens': 0, 'sverbtypes': {},

        'adjtypes': {}, 'adjtokens': 0,
        'advtypes': {}, 'advtokens': 0,
        'nountypes': {}, 'nountokens': 0,

        'lemmaposlist': [], 'lemmalist': []
    }

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

def process_lex_stats_coca(word, lemma, pos, result, wordranks):
    """

    n: noun
    v: verb
    j: adjective
    r: adverb

    :param word:
    :type word:
    :param lemma:
    :type lemma:
    :param pos:
    :type pos:
    :param result:
    :type result:
    :param wordranks:
    :type wordranks:
    :return:
    :rtype:
    """
    if pos not in string.punctuation and pos != "SYM":
        result['lemmaposlist'].append(pos)
        result['lemmalist'].append(lemma)
        result['wordtokens'] += 1
        result['wordtypes'][lemma] = 1
        if lemma not in wordranks[:2000] and (pos != "NN" or pos != "CD"):
            result['swordtypes'][lemma] = 1
            result['swordtokens'] += 1

        if pos[0] == "N":
            result['lextypes'][word] = 1
            result['nountypes'][word] = 1
            result['lextokens'] += 1
            result['nountokens'] += 1
            if lemma not in wordranks[:2000]:
                result['slextypes'][word] = 1
                result['slextokens'] += 1

        elif pos[0] == "J":
            result['lextypes'][word] = 1
            result['adjtypes'][word] = 1
            result['lextokens'] += 1
            result['adjtokens'] += 1
            if lemma not in wordranks[:2000]:
                result['slextypes'][word] = 1
                result['slextokens'] += 1

        elif pos[0] == "R":  # and (word in adjdict or (word[-2:] == "ly" and word[:-2] in adjdict)):
            result['lextypes'][word] = 1
            result['advtypes'][word] = 1
            result['lextokens'] += 1
            result['advtokens'] += 1
            if lemma not in wordranks[:2000]:
                result['slextypes'][word] = 1
                result['slextokens'] += 1

        elif pos[0] == "V" and word not in ["be", "have"]:
            result['verbtypes'][word] = 1
            result['verbtokens'] += 1
            result['lextypes'][word] = 1
            result['lextokens'] += 1
            if lemma not in wordranks[:2000]:
                result['sverbtypes'][word] = 1
                result['slextypes'][word] = 1
                result['slextokens'] += 1

    return result

def process_scores(i_filename, results):
    # adjust minimum sample size here
    standard = 50
    # 3.1 NDW, may adjust the values of "standard"
    ndw = ndwz = ndwerz = ndwesz = len(results['wordtypes'].keys())
    if len(results['lemmalist']) >= standard:
        ndwz = getndwfirstz(standard, results['lemmalist'])
        ndwerz = getndwerz(standard, results['lemmalist'])
        ndwesz = getndwesz(standard, results['lemmalist'])

    # 3.2 TTR
    msttr = ttr = len(results['wordtypes'].keys()) / float(results['wordtokens'])
    if len(results['lemmalist']) >= standard:
        msttr = getmsttr(standard, results['lemmalist'])

    verbtype_count = len(results['verbtypes'].keys())
    sophisticated_verb_count = len(results['sverbtypes'].keys())
    word_count = len(results['wordtypes'].keys())
    sword_count = len(results['swordtypes'].keys())
    wordtokens = results['wordtokens']
    swordtokens = results['swordtokens']
    lextokens = results['lextokens']
    slextokens = results['slextokens']

    if results['verbtokens'] == 0:
        print(f'WARNING: {i_filename} has zero verbtokens.')
        results['verbtokens'] = 1

    if results['wordtokens'] == 0:
        print(f'WARNING: {i_filename} has zero wordtokens.')
        results['wordtokens'] = 1

    if lextokens == 0:
        print(f'WARNING: {i_filename} has zero lextokens.')
        lextokens = 1

    if word_count == 0:
        print(f'WARNING: {i_filename} has zero word-count.')
        word_count = 1

    if wordtokens - ndw == 0:
        print(
            f'WARNING: {i_filename} will have a D of zero; wordtokens - ndw is zero. word_count artifically incremented by 1.')
        wordtokens = ndw + 1

    scores = {"filename": i_filename, "wordtypes": word_count, "swordtypes": sword_count,
              "lextypes": len(results['lextypes'].keys()), "slextypes": len(results['slextypes'].keys()),
              "wordtokens": wordtokens,
              "swordtokens": swordtokens, "lextokens": lextokens, "slextokens": slextokens,
              "ld": float(lextokens) / wordtokens,  # 1. lexical density
              # 2.1 lexical sophistication
              "ls1": slextokens / float(lextokens), "ls2": sword_count / float(word_count),

              # 2.2 verb sophistication
              "vs1": sophisticated_verb_count / float(results['verbtokens']),
              "vs2": sophisticated_verb_count * sophisticated_verb_count / float(results['verbtokens']),
              "cvs1": sophisticated_verb_count / sqrt(2 * results['verbtokens']),

              "ndw": ndw, "ndwz": ndwz, "ndwerz": ndwerz, "ndwesz": ndwesz, "ttr": ttr, "msttr": msttr,
              "cttr": word_count / sqrt(2 * wordtokens), "rttr": word_count / sqrt(wordtokens),
              "logttr": log(word_count) / log(wordtokens),
              "uber": (log(wordtokens, 10) * log(wordtokens, 10)) / log(wordtokens / float(word_count), 10),
              # 3.3 verb diversity

              "vv1": verbtype_count / float(results['verbtokens']),
              "svv1": verbtype_count * verbtype_count / float(results['verbtokens']),
              "cvv1": verbtype_count / sqrt(2 * results['verbtokens']),

              # 3.4 lexical diversity
              "lv": len(results['lextypes'].keys()) / float(lextokens),
              "vv2": len(results['verbtypes'].keys()) / float(lextokens),
              "nv": len(results['nountypes'].keys()) / float(results['nountokens']),
              "adjv": len(results['adjtypes'].keys()) / float(lextokens),
              "advv": len(results['advtypes'].keys()) / float(lextokens),
              "modv": (len(results['advtypes'].keys()) + len(results['adjtypes'].keys())) / float(lextokens),
              "D": (word_count ** 2) / (2 * (wordtokens - word_count))}

    if results['verbtokens'] == 0:
        scores['vv1'] = 0
        scores['vs2'] = 0
        scores['cvs1'] = 0
        scores['vs1'] = 0
        scores['svv1'] = 0
        scores['cvv1'] = 0

    if results['wordtokens'] == 0:
        scores['ld'] = 0
        scores['ls1'] = 0
        scores['ndw'] = 0
        scores['ndwz'] = 0
        scores['ndwerz'] = 0
        scores['ndwesz'] = 0
        scores['cttr'] = 0
        scores['uber'] = 0

    for key in scores.keys():
        if isinstance(scores[key], float):
            scores[key] = round(scores[key], 4)
    return scores

def process_spacy_syntax(spacy_syntax, word_count):
    count_dict = dict(Counter(spacy_syntax))
    w = word_count
    s = count_dict['ROOT']

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

    clause_keys = ['nsubj', 'nsubjpass', 'csubj', 'csubjpass']
    clause = 0
    for k in clause_keys:
        if k in count_dict:
            clause += count_dict[k]

    nmod_keys = ['nmod', 'npmod','tmod', 'poss']
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
    T = clause - (adverbclause + relcl + acl)
    VP = ccomp + clause
    passives = nsubjpass + csubjpass
    allmod = nmod + omod
    CSTR = (dc + xcomp + cord + nmod + omod)
    adjcl = acl + relcl
    mls = division(w, s)
    mlt = division(w, T)
    mlc = division(w, clause)
    c_s = division(clause, s)
    vp_t = division(VP, T)
    c_t = division(clause, T)
    t_s = division(T, s)
    co_s = division(cord, s)
    co_t = division(cord, T)
    co_c = division(cord, clause)
    adv_s = division(adverbclause, s)
    adv_t = division(adverbclause, T)
    adv_c = division(adverbclause, clause)
    adj_s = division(adjcl, s)
    adj_t = division(adjcl, T)
    adj_c = division(adjcl, clause)
    dc_s = division(dc, s)
    dc_t = division(dc, T)
    dc_c = division(dc, clause)
    pass_s = division(passives, s)
    pass_t = division(passives, T)
    pass_c = division(passives, clause)
    allmod_s = division(allmod, s)
    allmod_t = division(allmod, T)
    allmod_c = division(allmod, clause)
    CSTR_s = division(CSTR, s)
    CSTR_t = division(CSTR, T)
    CSTR_c = division(CSTR, clause)


    return {'w': w, 's': s, 'c': clause, 't-unit':T, 'vp':VP, 'ccomp':ccomp, 'xcomp':xcomp, 'cc': cord, 'advcl':adverbclause, 'acl':acl, 'relcl':relcl, 'adjcl':adjcl, 'nmod': nmod, 'omod': omod, 'allmod':allmod, 'dc': dc,  'pass':passives, 'CSTR':CSTR,
            'mls': mls, 'mlt':mlt, 'mlc': mlc, 'c_s': c_s, 'vp_t':vp_t, 'c_t':c_t, 't_s': t_s, 'co_s': co_s, 'co_t':co_t, 'co_c':co_c, 'adv_s':adv_s, 'adv_t':adv_t, 'adv_c':adv_c,
            'adj_s':adj_s, 'adj_t':adj_t, 'adj_c':adj_c, 'dc_s':dc_s, 'dc_t':dc_t, 'dc_c':dc_c, 'pass_s':pass_s, 'pass_t':pass_t, 'pass_c':pass_c, 'allmod_s':allmod_s, 'allmod_t':allmod_t, 'allmod_c':allmod_c, 'CSTR_s':CSTR_s, 'CSTR_t':CSTR_t, 'CSTR_c':CSTR_c}


def process_spacy(input_text, filename):
    """

    :param input_text:
    :param filename:
    :return:
    """
    spacy_results = prepare_empty_results()
    wordranks = read_coca_frequent_data()
    nlp = spacy.load("en_core_web_lg")
    spacy_tokens = nlp(input_text)
    spacy_syntax = []
    for idx, token in enumerate(spacy_tokens):
        spacy_word = spacy_tokens[idx].text
        spacy_tag = spacy_tokens[idx].tag_
        spacy_lemma = spacy_tokens[idx].lemma_
        spacy_syntax.append(spacy_tokens[idx].dep_)
        spacy_results = process_lex_stats_coca(spacy_word, spacy_lemma, spacy_tag, spacy_results, wordranks)

    spacy_scores = process_scores(filename, spacy_results)
    word_count = len([token for token in spacy_tokens if
                      token.is_alpha or token.shape_ == 'dd'])  # dd is spacy's definition for digits.
    spacy_syntax_results = process_spacy_syntax(spacy_syntax, word_count)
    return spacy_scores, spacy_syntax_results


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


class ArgCounter:
    """
    This handles the counting of words and phrases that appear in the list of supporting details markers.
    """
    def __init__(self, word_list_filepath):
        self.word_list = load_word_list(word_list_filepath)

    def count_arguments_substring(self, i_text):
        count = 0
        items = []
        text = i_text.lower()
        for word in self.word_list:
            if word in text:
                count += 1
                items.append(word)
        counts = dict(Counter(items))
        return count, counts

    def count_arguments_single(self, i_text):
        text = i_text.lower()

        text_words = re.split("\W+", text)
        details = {w: text_words.count(w) for w in self.word_list}
        count = sum(details.values())
        return count, details

    def count_arguments_regex_substring(self, i_text):
        text = i_text.lower()
        p = re.compile('|'.join(re.escape(w) for w in self.word_list))
        items = p.findall(text)
        details = dict(Counter(items))
        count = sum(details.values())
        return count, details

    def count_arguments(self, i_text):
        text = i_text.lower()
        regex_string = "|".join(rf"\b{re.escape(word)}\b" for word in self.word_list)
        regex = re.compile(regex_string, re.IGNORECASE)
        items = regex.findall(text)
        details = dict(Counter(items))
        count = sum(details.values())
        return count, details


def process(file_path: str, filename: str, arg_counter: ArgCounter):
    """

    :param file_path:
    :param filename:
    :param arg_counter:
    :return:
    """
    text_lines = read_input_text(file_path)
    input_text = ''.join(text_lines)
    spacy_scores, spacy_syntax_results = process_spacy(input_text, filename)
    arg_count, details = arg_counter.count_arguments(input_text)
    argument_scores = process_arguments(arg_count, spacy_syntax_results['w'])
    return {'scores': spacy_scores, 'syntax': spacy_syntax_results, 'argument_scores': argument_scores}


def main(input_path):
    input_filepath = os.path.join(os.getcwd(), input_path)
    mode = check_mode(input_filepath)
    ac = ArgCounter('argument_list.txt')

    if mode == 'file':
        result = process(input_path, Path(input_path).name, ac)
        print(f"Results for {result['scores']['filename']}")
        for k, v in result.items():
            print(f'{k}: {v}')

    if mode == 'directory':
        scores = []
        for fdx, filename in enumerate(os.listdir(input_filepath)):
            if filename.endswith('.txt'):
                result = process(os.path.join(input_filepath, filename), filename, ac)
                scores.append(result)

        header = build_header(scores)
        string_scores = list_stringify_scores(scores)
        write_header_and_data_to_file(header, string_scores, os.path.join(os.getcwd(),
                                                                          f'./output/spacy_full_out_{len(scores)}.csv'))




if __name__ == '__main__':
    assert sys.argv[1], 'input file parameter missing.'
    main(sys.argv[1])