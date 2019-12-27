# Created By Anthony Fuller
# 4/8/2019
# This program is to show how to create an error log file that can be read by the apps developer
# This program will also continually attempt to restart itself on failure
import time
import logging

def run():            
    while 1:
        try:
            ErrorLogExample()
        except:
            # Remove the break here to have the program restart automatically after an error
            break
            print("Restarting\n\n")
    pass

def ErrorLogExample():
    try:
        print("Dividing by zero as a test")
        print(1/0)

    except Exception as e:
        print(e)
        print("\nFAILURE TO INITIALIZE\n")
        logging.basicConfig(filename='ErrorLog.log',format='%(asctime)s %(message)s', level=logging.ERROR)
        # I included the tildes in order to help differenciate the longer error messages
        logging.debug(logging.exception("~~~~~~~~~~~~~~~~~~~~~~~~"))
        time.sleep(5)
        raise

if __name__ == "__main__":
    run()