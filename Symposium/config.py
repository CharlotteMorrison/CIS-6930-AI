import time
timestr = time.strftime("%Y%m%d-%H%M%S")
report = open("reports/report{}.txt".format(timestr), "w")
