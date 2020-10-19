import sys
from art import *

def mu2e_banner():
    return text2art('         Mu2e      Tracking')

def t_stage(var):
    return "[STAGE]: " + var

def t_system(var):
    return art('equalizer')+f"[{var}]"+art('equalizer')[::-1]

def t_mode(var):
    return '[MODE]: ' + var

def t_check_point(var, special=None):
    if special == None:
        return '[CHECK POINT]: \"'+var+ '\" is checked'
    else:
        return f'{special}[CHECK POINT]: \"'+var+ '\" is checked'

def t_debug(var, special=None):
    if special==None:
        return '[DEBUG]: '+f'{var}'
    else:
        return f'{special}[DEBUG]: '+f'{var}'

def t_warn(var, special=None):
    if special == None:
        return '[WARNING]: ' + var
    else:
        return f'{special}[WARNING]: ' + var

def t_error(var):
    return '[ERROR]: '+var

def t_info(var, special=None):
    if special==None:
        return '[INFO]: '+var
    else:
        return f'{special}[INFO]: '+var

### sys.stdout.writes
def pbanner():
    sys.stdout.write('='*102+'\n')
    sys.stdout.write(mu2e_banner()+'\n')
    sys.stdout.flush()

def psystem(var):
    sys.stdout.write(t_system(var)+'\n')
    sys.stdout.write('\n')
    sys.stdout.flush()

def pmode(var):
    sys.stdout.write(t_mode(var)+'\n')
    sys.stdout.flush()

def pstage(var):
    sys.stdout.write(t_stage(var)+'\n')
    sys.stdout.flush()

def pcheck_point(var, special = None):
    if special == None:
        text = t_check_point(var)+'\n'
    else:
        text = t_check_point(var, special)+'\n'
    sys.stdout.write(text)
    sys.stdout.flush()

def pdebug(var, message=None):
    sys.stdout.write(t_debug(var, message)+'\n')
    sys.stdout.flush()

def pwarn(var, special=None):
    sys.stdout.write(t_warn(var, special)+'\n')
    sys.stdout.flush()

def perr(var):
    sys.stdout.write(t_error(var)+'\n')
    sys.stdout.flush()

def pinfo(var, special=None):
    if special == None:
        sys.stdout.write(t_info(var)+'\n')
        sys.stdout.flush()
    else:
        sys.stdout.write(t_info(var, special=special)+'\n')
        sys.stdout.flush()
