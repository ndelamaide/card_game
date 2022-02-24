import numpy as np


def assign_card(rank, suit):
    
    card = None
    
    rank_to_string = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8',
                     9: '9', 10: 'J', 11: 'Q', 12: 'K', (-np.float('inf')): 'Null '}
    suite_to_string = {0: 'H', 1: 'D', 2: 'C', 3: 'S', None: 'Null'}
    
    if (suit in suite_to_string.keys()) & (rank in rank_to_string.keys()):
        
        card = rank_to_string[rank] + suite_to_string[suit]

    return card

def assign_cards(ranks, suits):
    
    cards = []
    
    for i in range(len(ranks)):
        cards.append(assign_card(ranks[i], suits[i]))
    
    return cards

def assign_points_std(ranks):
    
    points = []
    
    max_ = np.max(np.asarray(ranks))
    
    for i in range(len(ranks)):
        if ranks[i] == max_:
            points.append(1)
        else:
            points.append(0)
    
    return np.asarray(points)

def assign_points_adv(ranks, suits, dealer):
    
    points = []
    
    ranks_g = []

    # If we don't have suite of dealer, default to standard rules
    if suits[dealer-1] == None:
        return assign_points_std(ranks)
    
    else:
        for j in range(len(ranks)):
            if suits[j] == suits[dealer-1]:
                ranks_g.append(ranks[j])
        
        max_ = np.max(np.asarray(ranks_g))
        
        for i in range(len(ranks)):
            if suits[i] == suits[dealer-1]:
                if ranks[i] == max_:
                    points.append(1)
                else:
                    points.append(0)
            else:
                points.append(0)
                
        return np.asarray(points)