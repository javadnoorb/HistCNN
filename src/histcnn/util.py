import time
import sys
import os
from google.cloud import datastore
import pandas as pd
import subprocess
import pkg_resources

DATA_PATH = pkg_resources.resource_filename('histcnn', 'data/')

def read_csv(obj, columns=None, **kwargs):
    '''
    This function overloads pd.read_csv with the ability to read in more flexible input types
    
    arguments:
        obj (str or pd.DataFrame): if obj is a string (path) it uses it as
            a csv filename and applied pd.read_csv to it. If it is a pandas dataframe,
            it simply returns the dataframe intact.
        columns (list): if provided, will be used as the columns for the output of pd.read_csv().
            This option is only used if the input is read from a file.
        kwargs: any argument which is defined in pd.read_csv can be provided
    
    returns:
        pd.DataFrame 
        
    '''
    if type(obj) == pd.DataFrame:
        return obj
    elif type(obj) == str:
        df = pd.read_csv(obj, **kwargs)
        if columns is not None:
            df.columns = columns
        return df
    else:
        raise Exception('Unrecognized input.')

def run_unix_cmd(cmd, verbose=True):
    p = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, error = p.communicate()
    if verbose:
        print(output.decode())
    if p.returncode != 0: 
        raise Exception(error.decode())
    return output.decode()

def gsutil_cp(path1, path2, make_dir=False, verbose=True):
    if make_dir:
        mkdir_if_not_exist(path2)
    cmd = 'gsutil -m cp -r {:s} {:s}'.format(path1, path2)
    if verbose:
        print('copying {:s} -> {:s}'.format(path1, path2))
    output = run_unix_cmd(cmd, verbose=verbose)
    return output

def get_datastore_column(kind, key, project_id=None, uses_id=False):
    '''
    Get one (or a list of) column(s) from a datastore kind

    parameters:
        kind: str
            name of the datastore kind
        key: str, or list
            name of the column (or list of column names) to be fetched
        project_id: str, default None
            project id
        uses_id: bool, default False
            sometimes the entity names use id (instead of name). 
            This flag needs to be set to True in such cases
    returns:
        task_df: pandas.DataFrame
            a pandas dataframe containing the entities queried
    '''
    if not isinstance(key, list):
        key = [key]
    client = datastore.Client(project=project_id)

    query = client.query(kind=kind)
    all_tasks = list(query.fetch())
    tasks_df = pd.DataFrame(columns=key)
    for task in all_tasks:
        if uses_id:
            task_id = task.key.id
        else:
            task_id = task.key.name      
        for k in key:
            tasks_df.loc[task_id, k] = task.get(k)
    return tasks_df

def count_lines(filename):
    n = 0
    with open(filename) as fileobj:
        for line in fileobj:
            n+=1
    return n

def mkdir_if_not_exist(inputdir):
    if not os.path.exists(inputdir):
        os.makedirs(inputdir)
    return inputdir

class ProgressBar():
    def __init__(self,N,BarCounts = 40,newline_at_the_end=True, 
                    ProgressCharacter = "*", UnProgressCharacter = ' ', step = 1, dynamic_step = True):
        '''
        BarCounts : total number of bars in the progress bar
        newline_at_the_end : insert a newline when the loop ends
        ProgressCharacter : character to use for the progress bar (default is a full block)
        UnProgressCharacter : character to use for the remainder of progress bar (default is space)
        step : skip this many counts before updating the progress bar
        '''
        self.Time0 = time.time()
        self.BarCounts = BarCounts
        self.N = N
        self.i = 0
        self.newline_at_the_end = newline_at_the_end
        self.ProgressCharacter = ProgressCharacter
        self.UnProgressCharacter = UnProgressCharacter
        self.step = step    
        self.PrevWriteInputLength = 0
        self.dynamic_step = dynamic_step
        
    def Update(self,Text = '',no_variable_change = False, PrefixText=''):
        '''
        Text L: text to show during the update
        no_variable_change : do not update the internal counter if this is set to True
        '''        
        if not no_variable_change:
            self.i = self.i + 1
            
        if (self.i % self.step == 0) | (self.i==self.N):
            CurrentTime = (time.time()-self.Time0)/60.0
            CurrProgressBarCounts = int(self.i*self.BarCounts/self.N)
            self.WriteInput = u'\r%s|%s%s| %.1f%% - %.1f / %.1f minutes - %s'%(
                                                          PrefixText,
                                                          self.ProgressCharacter*CurrProgressBarCounts, 
                                                          self.UnProgressCharacter*(self.BarCounts-CurrProgressBarCounts), 
                                                          100.0*self.i/self.N, 
                                                          CurrentTime, 
                                                          CurrentTime*self.N/self.i,
                                                          Text)
            ExtraSpaces = ' '*(self.PrevWriteInputLength - len(self.WriteInput))    # Needed to clean remainder of previous text
            self.PrevWriteInputLength = len(self.WriteInput)                                        
            sys.stdout.write(self.WriteInput+ExtraSpaces)
            sys.stdout.flush()
            if (not no_variable_change) & self.newline_at_the_end & (self.i==self.N):
                print('\n')
    def NestedUpdate(self,Text = '',ProgressBarObj=None,no_variable_change = False, PrefixText=''): # nest this progressbar within another progressbar loop
        if ProgressBarObj!=None:
            ProgressBarObj.newline_at_the_end = False
            # assert not ProgressBarObj.newline_at_the_end, 'The object prints newline at the end. Please disable it.'
            self.newline_at_the_end = False
            no_variable_change = False
            PrefixText=ProgressBarObj.WriteInput
        self.Update(Text = Text,no_variable_change = no_variable_change, PrefixText=PrefixText)
       
