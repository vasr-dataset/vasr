places_clusters = {
            'inside_rooms': ['office', 'classroom', 'school', 'kitchen', 'bed', 'hallway', 'chair', 'toilet', 'bathroom', 'cafeteria', 'garage', 'library', 'bookshop'],
            'specific': ['church', 'concert', 'casino', 'dance floor',  'school', 'temple', 'palace', 'stage', 'golf course', 'horse race', 'wedding', 'hospital room', 'theater'],
            'outdoor_specific': ['gymnasium', 'farm', 'barn', 'restaurant', 'playing field', 'football field', 'football', 'football stadium', 'shrub', 'rink', 'skating rink', 'basketball court', 'grave'],
            'outdoor_street': ['door', 'park', 'parking lot', 'race track', 'car', 'railway station', 'plaza', 'bridge', 'ski run', 'ski trail'],
            'outdoor_nature': ['trail', 'rock', 'snow', 'cliff', 'park'],
            'outdoor_nature_ambiguous': ['forest', 'mountain', 'hill'],
            'outdoor_water': ['ocean', 'body of water', 'lake', 'river', 'water', 'pond', 'sea', 'beach', 'pool', 'shore', 'beach', 'wetland', 'white water']
}

general_synsets = ['animal', 'person', 'group', 'male', 'female', 'creation', 'wheeled_vehicle', 'system_of_measurement',
                           'structure', 'phenomenon', 'covering', 'celestial_body', 'food', 'furniture', 'body_of_water',
                           'instrumentality', 'geographical_area', 'round_shape', 'plant', 'fire', 'tube', 'educator',
                           'liquid', 'leaf', 'figure', 'substance', 'volcanic_eruption', 'natural_elevation', 'force', 'bird_of_prey', 'bovine', 'skeleton',
                           'male', 'female', 'body_part', 'conveyance', 'utensil', 'dog', 'cat', 'rock', 'hoop', 'way', 'horseman',
                           'spiritual_leader', 'spring', 'doll', 'plant_part', 'piece_of_cloth', 'piece_of_cloth', 'plant_organ', 'edible_fruit', 'cord',
                           'jewelry', 'baseball', 'poster', 'javelin', 'cement', 'fabric', 'snow', 'football', 'ice', 'tape', 'screen', 'grave', 'plate',
                           'plastic', 'egg', 'collar', 'ribbon', 'rope', 'wool', 'glass', 'lumber', 'cake', 'powder', 'sink', 'balloon',
                           'mushroom']


human_keys = {'people', 'person', 'man', 'woman', 'male_child', 'female_child', 'group', 'worker', 'entity', 'family', 'couple', 'crowd', 'marriage', 'group', 'adolescent', 'girl', 'boy'}


blacklist_individuals_ambiguous = ['standing', 'walking', 'recuperating', 'cresting', 'misbehaving', 'rehabilitating',
                                   'stpooping', 'cresting']
wadling_cluster = ['waddling', 'walking', 'running', 'jogging']
arrest_cluster = ['apprehending', 'detaining', 'handcuffing', 'frisking', 'tackling', 'bandaging']
sick_cluster = ['recuperating', 'coughing']
sprinting_cluster = ['sprinting', 'racing', 'running', 'skidding']
cheering_cluster = ['cheering', 'whirling', 'dancing']
weeping_cluster = ['gasping', 'weeping', 'grieving']
stretching_cluster = ['stretching', 'wringing', 'rehabilitating']
mashing_cluster = ['mashing', 'shredding']
smiling_cluster = ['smiling', 'laughing', 'grinning']
raining_cluster = ['raining', 'snowing', 'storming']
nipping_cluster = ['nipping', 'tugging']
parading_cluster = ['marching', 'parading', 'protesting']
slipping_cluster = ['stumbling', 'stipping']
kneeling_cluster = ['foraging', 'kneeling']
swooping_cluster = ['swooping', 'soaring']

whitelist_pairs = [['biking', 'hitchhiking']]


blacklist_pairs = [['truck', 'trailer'], ['jaguar', 'leopard'], ['car', 'car door'], ['racer', 'rider'], ['cruiser', 'truck'], ['cruiser', 'tractor'], ['preacher', 'priest'], ['racer', 'runner'],['light', 'light_bulmb'], ['glass', 'liquor'],
                   ['glass', 'people'], ['glass', 'inside'], ['gull', 'eagle'], ['hawk', 'gull'], ['hawk', 'eagle'], ['car', 'racer'], ['racer', 'motorcycle'], ['group', 'child'], ['preacher', 'curate'], ['athlete', 'team'], ['person', 'horse'],
                   ['priest', 'curate'], ['hand', 'finger'], ['arm', 'hand']]