import player_tracking_ss_v1

match_details = [[901, 4032000, 4040000, 25], [904, 6537000, 6547000, 30],[906, 526000, 531960, 25], [907, 77000, 81960, 25], [908, 61680, 67040, 25]]
# match_details = [[904, 6537000, 6547000, 30]]
# match_details = [[906, 92000, 100000, 25], [901, 1667000, 1676000, 25]] # Extra vids to test if possible

player_tracking_ss_v1.run_player_tracking_ss(match_details, False)