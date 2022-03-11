from twitter_watcher import *

if __name__ == "__main__":
    watcher = Twitter_Watcher("./data/data.csv")
    watcher.run_event_search()