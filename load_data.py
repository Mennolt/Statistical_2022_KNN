import numpy as np

def load_data(loc : str,**kwargs) -> np.ndarray:
    """
    Loads a minst datafile and converts it into an numpy ndarray
    :param loc: location of the datafile
    :return: the data as an ndarray
    """
    data = np.genfromtxt(loc, delimiter=",")
    out = Scale.control(data,**kwargs)
    return out

class Scale:
    #This controls the logic for converting the data from 0-255 to 0-1.
    #There are a couple keywords to use this to send in kwargs.
    #Keywords
    #   Scale sets whether its used or not
    #   Val sets the value for the cut off, if not used than it will choose the 90 percentile of all non zero elements
    #   Percent which sets the cut off based on all non zero elements (default 90)
    def control(out,**kwargs):
        if 'scale' in kwargs.keys() and kwargs['scale']==True:
            if 'val' not in kwargs.keys():
                out = Scale.val_approximater(out,**kwargs)
            else:
                out = Scale.bool_scaler(out,kwargs['val'])
        return(out)
    
    #Approximates the cut off for 0 based on an inputted percent or through the default 90
    #The cut off is based solely on the percentiles of all non zero elements
    def val_approximater(data,**kwargs):
        data_wo_first = data[:, 1:]
        non_zero_arr = data_wo_first[np.nonzero(data_wo_first)]
        if 'percent' in kwargs.keys():
            percent = kwargs['percent']
        else:
            percent = 90
        print(np.mean(non_zero_arr))
        cut_off = np.percentile(non_zero_arr,percent)
        print(cut_off)
        out = Scale.bool_scaler(data,cut_off)
        return(out)

    #Computes actual 0 and 1 setter
    def bool_scaler(data, val):
        data = np.where(data < val, 0, 1)
        return(data)