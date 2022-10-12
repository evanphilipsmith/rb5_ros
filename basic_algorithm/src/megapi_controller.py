from megapi import MegaPi


MFR = 2     # port for motor front right
MBL = 3     # port for motor back left
MBR = 10    # port for motor back right
MFL = 11    # port for motor front left


class MegapiController:
    def __init__(self, port='/dev/ttyUSB0', verbose=True, dryrun=False):
        self.port = port
        self.verbose = verbose
        self.dryrun = dryrun
        if verbose:
            print("Communication Port: {}".format(port))
            print("Motor ports:\n\tFR: {}\tBL: {}\t BR: {}\tFL: {}".format(MFR, MBL, MBR, MFL))
        self.bot = MegaPi()
        self.bot.start(port=port)
        self.mfr = MFR  # front right motor port
        self.mbl = MBL  # back left motor port
        self.mbr = MBR  # back right motor port
        self.mfl = MFL  # front left motor port

    def setMotors(self, front_left=0, front_right=0, back_left=0, back_right=0):
        if self.verbose:
            print("FL: {}\tFR: {}\tBL: {}\tBR: {}".format(
                int(round(front_left, 0)),
                int(round(front_right, 0)),
                int(round(back_left, 0)),
                int(round(back_right, 0))))
        if self.dryrun:
            return
        self.bot.motorRun(self.mfl, front_left)
        self.bot.motorRun(self.mfr, front_right)
        self.bot.motorRun(self.mbl, back_left)
        self.bot.motorRun(self.mbr, back_right)

    def stop(self):
        if self.verbose:
            print("STOP")
        self.setMotors()
    
    def close(self):
        self.bot.close()
        self.bot.exit()
