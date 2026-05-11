import warnings
import numpy as np
from qed.utils.sec_mole import read_molecule

def read_keyword_block(data):
    r"""Read keyword block like $rem ... $end."""
    rem_keys = {}
    for line in data:
        if '!' not in line:
            info = line.split()
        elif len(line.split('!')[0]) > 0:
            info = line.split('!')[0].split()
        else:
            info = []

        if len(info) == 2:
            rem_keys[info[0].lower()] = convert_string(info[1])
        elif len(info) > 2:
            rem_keys[info[0].lower()] = [convert_string(x) for x in info[1:]]

    #print('rem_keys: ', rem_keys)
    return rem_keys


def convert_string(string):
    r"""Convert string to int, float, or keep as string."""
    if string.isdigit():
        return int(string)
    else:
        try:
            return float(string)
        except:
            return string


def parser(file_name):
    r"""Parse input file into dictionary of parameters."""
    infile = open(file_name, 'r')
    lines = infile.read().split('$')
    #print('lines:\n', lines)

    parameters = {}
    for section in lines:
        data = section.split('\n')
        name = data[0].lower()
        #function = 'read_' + name
        #if function in globals():
        #    parameters[name] = eval('read_'+name)(data)
        if name == 'molecule':
            parameters[name] = read_molecule(data)
        else:
            parameters[name] = read_keyword_block(data)

    print('parameters:\n', parameters)
    return parameters



def put_keys_kwargs_to_object(obj, key={}, **kwargs):
    r"""
    Put the individual keys in the dictionary first,
    then use dictionary to assign the obj class attributes.
    """
    keys = put_kwargs_to_keys(key, **kwargs)
    put_kwargs_to_object(obj, **keys)


def put_kwargs_to_keys(key={}, **kwargs):
    r"""Put all the individual keyword and value into the key dictionary."""
    for name, value in kwargs.items():
        if name in key.keys():
            warnings.warn('keyword %s is over written as %s' % (name, value), DeprecationWarning)
        key[name] = value

    return key


def put_keys_to_kwargs(key, **kwargs):
    r"""Put the values in key dict to kwargs dict."""
    for name, value in key.items():
        if name in  kwargs.keys():
            warnings.warn('keyword %s is over written as %s' % (name, value), DeprecationWarning)
        kwargs[name] = value

    return kwargs


def put_kwargs_to_object(obj, **kwargs):
    r"""Put all the keywords and values into the obj class."""
    class_variables = kwargs.pop('class_variables', None)
    if class_variables:
        for var in class_variables:
            for name, value in kwargs.items():
                if name == var:
                    setattr(obj, name, value)

    else:
        for name, value in kwargs.items(): # put all the variables in the class
            if isinstance(value, list):
                try: # try to convert to numpy array
                    value = np.array(value)
                except:
                    pass
            setattr(obj, name, value)
        #else:
        #    raise Exception('%s class does not have %s attribute' % (obj.__class__.__name__, name))


def put_keys_to_object(obj, key):
    r"""Put all the keywords and values into the obj class."""
    put_kwargs_to_object(obj, **key)



if __name__ == '__main__':
    import sys
    infile = 'water.in'
    if len(sys.argv) >= 2: infile = sys.argv[1]
    parameters = parser(infile)
