def recommend(draftState, user_team_idx):
    return

def pos_select(draft_state, user_team_idx):
    user_team = draft_state.team_list[user_team_idx]
    needs, bench_remaining = user_team.get_needs()
    if(len(needs) == 1):
        if(needs[0] == "FLEX"):
             needs.append("WR")
             needs.append("TE")
             needs.append("RB")
             needs.remove("FLEX")
             scarcity = {pos : get_scarcity(draft_state, pos) for pos in needs}
             return max(scarcity, key=scarcity.get) #type: ignore
        else:
            return needs[0]
    elif(len(needs) > 1):
        if("FLEX" in needs):
             needs.append("WR")
             needs.append("TE")
             needs.append("RB")
             needs.remove("FLEX")
             needs = set(needs)
        scarcity = {pos : get_scarcity(draft_state, pos) for pos in needs}
        return max(scarcity, key= scarcity.get)  #type: ignore
    else:
        return

    
    return

def get_pick_gap(round, pick_number, n_teams):
        if(round % 2 ==0):
             current_pick = abs(n_teams - 1 - pick_number)
             next_pick = pick_number
        else:
             current_pick = pick_number
             next_pick = abs(n_teams - 1 - pick_number)

        return (n_teams - pick_number - 1) + abs(n_teams - 1 - pick_number)

def get_scarcity(draftstate, pos):
    positinal_starter_count=0
    for teams in draftstate.team_list:
          if(pos == "WR"):
               positinal_starter_count += (teams.wr_ct - len(teams.wr_list))
          elif(pos == "RB"):
                positinal_starter_count += (teams.rb_ct - len(teams.rb_list))
          elif(pos == "QB"):
               positinal_starter_count += (teams.qb_ct  - len(teams.qb_list))
          elif(pos == "TE"):
               positinal_starter_count += (teams.te_ct - len(teams.te_list))
    highest_tier = draftstate.available_players[draftstate.available_players["position"] == pos]["tier"].min()
    supply = len(draftstate.available_players[
        (draftstate.available_players["tier"] == highest_tier) & 
        (draftstate.available_players["position"] == pos)])
    scarcity = positinal_starter_count/supply
    return scarcity



  

def player_select(user_roster, available_players):
    return