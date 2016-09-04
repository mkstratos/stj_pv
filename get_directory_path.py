import numpy as np
import scipy.io as io
import pdb
import os				

__author__ = "Penelope Maher" 

#file purpose: To set up dir for input/output of data and plots


class Directory:
  'Working directory information and set precision'

  def __init__(self):

    self.base =  os.environ['BASE']
    
    np.set_printoptions(linewidth=150,precision=3,suppress=True)

    if self.base == '/home/z3319090/hdrive':
      self.work_loc = 'UNSW_server'
      self.data_loc = os.environ['DATA_15'] + 'Chain_Data/'

    if self.base == '/home/maher/Documents/Penny':
      self.work_loc = 'PeronalLaptop'
      self.data_loc = '/media/Seagate Expansion Drive/Data/' 


    if self.base == '/home/pm366/Documents':
      self.work_loc = 'ExeterLaptop' 
      self.data_loc = '/media/pm366/Seagate Expansion Drive/Data/'

    if self.base == '/home/links/pm366/Documents/':    
      self.work_loc = 'gv' 
      self.data_loc = '/scratch/pm366/' 

    self.plot_loc = self.base +'/Plots/'

 
    #test if the string self.work_loc has been assigned
    assert isinstance(self.work_loc, str), 'Unknown base environment variable. SetDefault.py'

def GetDiri():

  
  np.set_printoptions(linewidth=150,precision=3,suppress=True)

  Dir = False
  diri = {}
  base = os.environ['BASE']
  diri['base'] = base
  
  #working on home laptop  
  if base == '/home/maher/Documents/Penny':
    work_loc = 'PeronalLaptop'
    Dir = True 
    diri['data'] = '/media/Seagate Expansion Drive/Data/' 

  if base == '/home/pm366/Documents':
    work_loc = 'ExeterLaptop' 
    Dir = True
    diri['data_seagate'] = '/media/pm366/Seagate Expansion Drive/Data/'
    diri['data_laptop_disk'] = '/home/pm366/Documents/Data/'
    
    
  if base == '/home/links/pm366/Documents/':    
    work_loc = 'gv2' 
    Dir = True
    diri['data'] = '/scratch/pm366/' 
    
    
  if  Dir == False :
    print 'Unknown file location: open file SetDefaults and input new work location'
    pdb.set_trace()

  #diri['base'] = base + '/Data/'
    
  diri['base_plt'] = base +'/Plots/'
  diri['work_loc']=work_loc   

  print 'work_loc:', work_loc
  print 'base path:', base
  
  
  return diri
    
