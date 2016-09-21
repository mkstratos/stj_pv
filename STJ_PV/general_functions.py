from netCDF4 import Dataset
import numpy.ma as ma
import pickle
import numpy as np
import pdb


def openNetCDF4_get_data(filename):
  'Key names are in unicode'
  var = {}
  f = Dataset(filename, mode='r')
  print('opened: ',filename)

  var_names = list(f.variables.keys())
 
  for i in var_names:
     var[i]=f.variables[i][:] 

  f.close()

  return var

def apply_mask_inf(data):


    assert  isinstance(data,np.ndarray)  == True , 'Data is not masked'
    shape_data=data.shape
    data=data.flatten()
    wh = np.where(np.isfinite(data) == False)[0]
    if len(wh) != 0:
      data[wh] = np.nan

    mask_out=ma.masked_array(data,mask=np.isnan(data))
    mask_out=mask_out.reshape(shape_data) 
    
    return mask_out

def apply_mask_num(mask,data,num,num2=None):
    'num (0 not in MDB, 1 is in MDB) and num2 is the missing value'

    shape_mask=mask.shape
    shape_data=data.shape
    mask=mask.flatten().astype(float)
    data=data.flatten()
    mask_wh=np.where(mask == num)[0] 		#where mask is zero
    mask[mask_wh]=np.nan			#put nans on elements that are 0
  
    if num2 != None:
      #mask where missing data 
      mask_wh_miss=np.where(data == num2)[0]	
      mask[mask_wh_miss]=np.nan			#put nans on missing values
      
    non_triv=np.where(np.isfinite(mask) == True)[0].shape  #just for reference
    
    mask_out=ma.masked_array(data,mask=np.isnan(mask))
    mask_out=mask_out.reshape(shape_data) 
 
    return mask_out.reshape(shape_mask)

def addToList(num,list_data):
    #only add to a list if its unique
    for elem in range(len(num)):
      if num[elem] not in list_data:
        list_data.append(num[elem])

def save_file(filename, data, var_name, dim,var_type, dim_name,var_dim_name, specific=None,history=None):
 
    f=io.netcdf.netcdf_file(filename, mode='w')

    for j in range(dim):
      if dim_name[j] == 'missing_value':
        f.createDimension(dim_name[j],1) 
      else:
	f.createDimension(dim_name[j],len(data[dim_name[j]])) 

    for i in range(len(var_name)):
 
 	if var_name[i]=='history':
	  f.history = history	
	else:
          tmp = f.createVariable(var_name[i],var_type[i],var_dim_name[i])
	  #print var_name[i]
          tmp[:] =  data[var_name[i]]

	
        if specific == 'corr_index':
  	  if i == 0 :
	    tmp.indices_name=['STRI-STRP, STJI-STRI,STJP-STRP,STJI-STJP']
	    tmp.tau_names=['14400, 28800, T14400-Q28800, Q14400-T28800']
        if specific == 'conv2ls':
          if i ==0:
	    tmp.tau_control =['14400','28800']
          if i ==2:
	    tmp.tau_2_tau =['T14400-Q28800,T14400-Q28800']	
	     		

    f.close()    

    print('created file: ',filename)

    return f


def openfile_get_data(filename):
  
  var ={}
  f=open_file(filename)
  var_names = list(f.variables.keys())
  
  for i in var_names:
     var[i]=f.variables[i].data 

  f.close()
  return var 
  
def MeanOverDim(data,dim):
  'Assumed data is masked with nans for missing values - if inf then mask first'
  return np.nansum(data,axis=dim)/np.sum(np.isfinite(data),axis=dim)

def FindClosest(in_val, in_list):
  'Find the element of in_list that is closest to the value of in_val.'
  closest=lambda num,collection:min(collection,key=lambda x:abs(x-num))

  return closest(in_val,in_list)

def FindClosestElem(in_val, in_list):
  'Find the location of the elements in in_list that is closest to in_val'
  value   = FindClosest(in_val=in_val, in_list=in_list)
  element = np.where(value == in_list)[0]

  #print value, element
  
  if len(element) != 1 :
      print('Cant find closest element')
      pdb.set_trace()
    
  return element

def OpenPickle(filenamePickle):
  
  data = pickle.load( open(filenamePickle, "rb" ) )

  return data

def SavePickle(filenamePickle,data):
  
  pickle.dump(data, open(filenamePickle, "wb" ) )
  print('Pickle saved')  
  return ()

#this code ws adapted from stack overflow suggestion:
#http://stackoverflow.com/questions/28681782/create-table-in-subplot
def latex_table(celldata,rowlabel,collabel):
    'function that creates latex-table'

    table = r'\begin{tabular}{c|'''
    for c in range(len(collabel)):
        # add additional columns
        table += r'c|'
    table += r'} '

    # provide the column headers
    for c in range(len(collabel)-1):
        table += collabel[c]
        table += r' & '
    table += collabel[-1]
    table += r'\\ \hline '

    # populate the table:
    # this assumes the format to be celldata[index of rows][index of columns]
    for r in range(len(rowlabel)):
        table += rowlabel[r]
        table += r' & '
        for c in range(0,len(collabel)-2):
            if not isinstance(celldata[r][c], str):
                table += str(celldata[r][c])
            else:
                table += celldata[r][c]
            table += r' & '

        if not isinstance(celldata[r][-1], str):
            table += str(celldata[r][-1])
        else:
            table += celldata[r][-1]
        table += r'\\  '

    table += r'\end{tabular} '


    return table



