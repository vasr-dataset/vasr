from collections import defaultdict

from nltk.corpus import wordnet as wn

from dataset.utils.wordnet_consts import general_synsets, human_keys, sick_cluster, arrest_cluster, \
    mashing_cluster, cheering_cluster, nipping_cluster, stretching_cluster, raining_cluster, smiling_cluster, \
    wadling_cluster, sprinting_cluster, parading_cluster, weeping_cluster, slipping_cluster, kneeling_cluster, \
    swooping_cluster, blacklist_individuals_ambiguous

empty_clusters = []
final_reject = []


class PairsFilter(object):
    def __init__(self):
        relevant_keys = ['diff_item_A_str_first', 'diff_item_B_str_first']
        relevant_keys_wn = ['diff_item_A', 'diff_item_B']
        merged_keys = relevant_keys + relevant_keys_wn
        self.keys = merged_keys
        self.specific_synsets_dict = {}
        for s in general_synsets:
            self.specific_synsets_dict[s] = wn.synsets(s)

        self.filter_counts = defaultdict(int)

    def print_filter_stats(self):
        sum_filter_stats = sum(self.filter_counts.values())
        sum_filter_stats_A = sum([v for k,v in self.filter_counts.items() if 'A_' in k])
        sum_filter_stats_R = sum([v for k,v in self.filter_counts.items() if 'R_' in k])
        print(f"*** print_filter_stats ***")
        print(f"Total: {sum_filter_stats}, Accept: {sum_filter_stats_A}, Reject: {sum_filter_stats_R}")
        print(self.filter_counts)
        normalized = {k:str(round(v / sum_filter_stats * 100, 1))+"%" for k,v in self.filter_counts.items()}
        from pprint import pprint
        pprint(normalized)


    def is_legit_k_chagnge(self, k, t):
        if k == 'verb':
            return self.is_legit_verb_change(t)
        else:
            return self.is_legit_object_change(t)

    def is_legit_object_change(self, t):
        t_dict = dict(zip(self.keys, t))
        wn_x = wn.synset_from_pos_and_offset('n', int(t_dict['diff_item_A'].split("n")[1]))
        wn_y = wn.synset_from_pos_and_offset('n', int(t_dict['diff_item_B'].split("n")[1]))
        x_name = wn_x.lemmas()[0].name()
        y_name = wn_y.lemmas()[0].name()
        lowest_common_hypernyms = wn_x.lowest_common_hypernyms(wn_y)
        try:
            x_clusters = self.get_clusters(wn_x, x_name)
            y_clusters = self.get_clusters(wn_y, y_name)
        except Exception as ex:
            print("Problems with clusters, returning False")
            print(t_dict)
            self.filter_counts['R_error_cluster'] += 1
            return False
        xy_clusters = x_clusters + y_clusters
        xy_words = [x_name, y_name]

        if x_clusters == y_clusters and len({'animal'}.intersection(xy_clusters)) == 0 \
                and len(lowest_common_hypernyms) > 0 and len(human_keys.intersection(xy_words)) == 0:
            abstract_class = lowest_common_hypernyms[0]
            abstract_class_name = abstract_class.name().split(".")[0]
            abstract_names_blacklist = human_keys.union({'whole', 'vascular_plant', 'woody_plant'})
            if abstract_class_name not in abstract_names_blacklist and abstract_class_name not in xy_words and \
                    has_hypernym_wn_synset_inner(wn_x, abstract_class) and \
                    has_hypernym_wn_synset_inner(wn_y, abstract_class) and not \
                    objects_are_type_of_each_other(x_name, y_name) and not (x_name in y_name or y_name in x_name):
                self.filter_counts['A_instances'] += 1
                return True
        if 'animal' in xy_clusters and not objects_are_type_of_each_other(x_name, y_name):
            if ('dog' in xy_clusters and 'puppy' in xy_words) or ('cat' in xy_clusters and 'kitten' in xy_words):
                self.filter_counts['R_same_animal'] += 1
                return False
            self.filter_counts['A_animal'] += 1
            return True
        if 'food' in xy_clusters and not objects_are_type_of_each_other(x_name, y_name):
            self.filter_counts['A_food'] += 1
            return True
        if ('wheeled_vehicle' in xy_clusters or 'conveyance' in xy_clusters) and not objects_are_type_of_each_other(x_name, y_name):
            self.filter_counts['A_wheeled_vehicle'] += 1
            return True
        if 'body_part' in xy_clusters and len(set(xy_clusters)) == 1:
            self.filter_counts['A_body_part'] += 1
            return True
        if 'male' in xy_clusters and 'female' in xy_clusters:
            self.filter_counts['A_male_to_female'] += 1
            return True
        if len(human_keys.intersection(xy_clusters)) == 0 and len(set(x_clusters).intersection(y_clusters)) == 0:
            ''' If no human key, and different clusters - it's ok'''
            self.filter_counts['A_different_clusters_no_human'] += 1
            return True
        if len(set(x_clusters).intersection(y_clusters)) == 0 and len(human_keys.intersection(xy_clusters)) == 1:
            ''' If different clusters, we may allow 1 human key '''
            for human_key in human_keys:
                if human_key in xy_clusters:
                    other_obj_candidate = y_name if human_key in x_clusters else x_name
                    if not objects_are_type_of_each_other(human_key, other_obj_candidate):
                        self.filter_counts['A_different_clusters_one_human_key'] += 1
                        return True
        self.filter_counts['R_final'] += 1
        t_dict['x_name'] = x_name
        t_dict['y_name'] = y_name
        global final_reject
        final_reject.append(t_dict)
        return False

    def is_legit_verb_change(self, t):
        t_dict = dict(zip(self.keys, t))
        if t_dict['diff_item_A'] == t_dict['diff_item_B']:
            return False
        blacklist_clusters = [sick_cluster, arrest_cluster, mashing_cluster, cheering_cluster,
                              smiling_cluster, raining_cluster, stretching_cluster, nipping_cluster,
                              weeping_cluster, parading_cluster, sprinting_cluster, wadling_cluster,
                              slipping_cluster, kneeling_cluster, swooping_cluster]
        whitelist_pairs = [['biking', 'hitchhiking']]
        xy_verbs = {t_dict['diff_item_A'], t_dict['diff_item_B']}
        for p in whitelist_pairs:
            if len(set(p).intersection(set(xy_verbs))) == 2:
                return True
        for c in blacklist_clusters:
            if len(xy_verbs.intersection(set(c))) >= 2:
                return False
        if len(xy_verbs.intersection(set(blacklist_individuals_ambiguous))) >= 1:
            return False
        x_wns = [x for x in wn.synsets(t_dict['diff_item_A']) if x.pos() == 'v']
        y_wns = [x for x in wn.synsets(t_dict['diff_item_B']) if x.pos() == 'v']
        if len(y_wns) == 0 or len(x_wns) == 0:
            return False
        wn_x = x_wns[0]
        wn_y = y_wns[0]
        xy_sim = wn_x.wup_similarity(wn_y)
        if xy_sim > 0.5:
            return False
        return True

    def get_clusters(self, wn_x, x_str):
        clusters = self.special_cases(x_str)
        if len(clusters) == 0:
            clusters = self.get_clusters_inner(wn_x)
            if len(clusters) == 0:
                global empty_clusters
                empty_clusters.append(x_str)
        return clusters

    def special_cases(self, x_str):
        clusters = []
        if x_str.lower() in ['duckling', 'yak', 'tiger', 'leopard', 'alligator']:
            clusters = ['animal']
        if x_str.lower() in ['labrador', 'pug', 'puppy']:
            clusters = ['dog', 'animal']
        if x_str.lower() in ['cat', 'kitten']:
            clusters = ['cat', 'animal']
        if x_str.lower() in ['fireman', 'driver', 'racer', 'protest', 'marriage']:
            clusters = ['person']
        if x_str.lower() in ['professor']:
            clusters = ['teacher']
        if x_str.lower() in ['wave', 'spring', 'hot_spring']:
            clusters = ['body_of_water']
        if x_str.lower() in ['ember', 'log']:
            clusters = ['wood']
        if x_str.lower() in ['crop']:
            clusters = ['plant']
        if x_str.lower() in ['grass', 'field']:
            clusters = ['field']
        if x_str.lower() in ['police', 'policeman', 'detective', 'motorcycle_cop', 'motorcycle cop'] or ('_' in x_str and 'cop' in x_str.split("_")):
            clusters = ['person', 'force']
        if x_str.lower() in ['serviceman', 'teamster']:
            clusters = ['person', 'worker', 'force']
        if x_str.lower() in ['soldier', 'military personnel', 'marching_band', 'band', 'army', 'security']:
            clusters = ['force', 'people']
        return clusters

    def get_clusters_inner(self, wn_x):
        s_clusters = []
        for s, s_syn_lst in self.specific_synsets_dict.items():
            for s_syn in s_syn_lst:
                if has_hypernym_wn_synset_inner(wn_x, s_syn):
                    s_clusters.append(s)
        return s_clusters


def has_hypernym_wn_synset_inner(word_synset, category_synset):
    for match in word_synset.lowest_common_hypernyms(category_synset):
        if match == category_synset:
            return True
    return False

def has_hypernym_wn_synset(word_synset, category_synset):
    return has_hypernym_wn_synset_inner(word_synset, category_synset) or has_hypernym_wn_synset_inner(category_synset, word_synset)


def objects_are_type_of_each_other(x, y):
    for syn_x in wn.synsets(x):
        for syn_y in wn.synsets(y):
            if has_hypernym_wn_synset(syn_x, syn_y):
                return True
    return False

