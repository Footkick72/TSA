import pstats

stats = pstats.Stats("profilestats")
stats.sort_stats("tottime")
stats.print_stats()