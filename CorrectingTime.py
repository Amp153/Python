# Created By Anthony Fuller
# 4/9/2019
# This program is to show how to syncronize the time by the milisecond on a windows machine to binance
# Make sure that it's run with administrative privlages since it's resyncing the system clock.
# You need to call set_time_binance() right before every transaction because it'll be off after a couple of minutes
# I'm worried about an NDA I'm under so I'm giveing the bare minimum
import time
import win32api
import urllib.request

# client = Client(api_key, api_secret)

def run():            
    while 1:
        try:
            # This is to check if you can connect to binance in the first place
            while urllib.request.urlopen("http://api.binance.com").getcode() != 200:
                        print("Lost connection to the server. Retrying in 30 seconds")
                        time.sleep(30)
            set_time_binance()
        except:
            # Remove the break here to have the program restart automatically after an error
            break
            print("Restarting\n\n")
    pass

def set_time_binance():
    gt=client.get_server_time()
    tt=time.gmtime(int((gt["serverTime"])/1000))
    win32api.SetSystemTime(tt[0],tt[1],0,tt[2],tt[3],tt[4],tt[5],0)

if __name__ == "__main__":
    run()