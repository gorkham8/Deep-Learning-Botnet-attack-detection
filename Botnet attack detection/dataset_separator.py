import numpy as np
import pandas as pd
import glob
import os 

n_BaIoT_dataset = pd.DataFrame(index=range(0), columns=range(1))
reset = n_BaIoT_dataset

def label_adder(x,label_code):
    column = pd.DataFrame(index=range(x),columns=range(0))
    column.at[0:x,'result']= label_code
    return column  

def device_id_adder(x,label_code):
    column = pd.DataFrame(index=range(x),columns=range(0))
    column.at[0:x,'device_id']= label_code
    return column  

def merge_datasets(incoming_dataset):
    global n_BaIoT_dataset
    n_BaIoT_dataset = pd.concat([n_BaIoT_dataset, incoming_dataset])
    print(1)



def main():
    
    a = 0
    for name in glob.glob('C:/Users/User/Spyder Warehouse/n-BaIoT/*benign.csv'):
        # print("folder")
        df = pd.read_csv(name)  
        x = len(df)
        print(x)
        col_1=device_id_adder(x,a)
        df['device_id'] = col_1
        
        col_2=label_adder(x,0)
        df['result'] = col_2
        if (a==0):
            global n_BaIoT_dataset
            n_BaIoT_dataset = df.copy(deep=True)
            print(2)
        else:
            merge_datasets(df)
    
        a=a+1
        
    n_BaIoT_dataset.to_csv('n_BaIoT_0.csv',index=False)
    n_BaIoT_dataset = reset
        
    a = 0
    for name in glob.glob('C:/Users/User/Spyder Warehouse/n-BaIoT/*combo.csv'):
        print("combo folder")
        df = pd.read_csv(name)  
        x = len(df)
        print(x)
        col_1=device_id_adder(x,a)
        df['device_id'] = col_1
        
        col_2=label_adder(x,1)
        df['result'] = col_2
        if (a==0):
            n_BaIoT_dataset = df.copy(deep=True)
        else:
            merge_datasets(df)
        
        a=a+1
        
    n_BaIoT_dataset.to_csv('n_BaIoT_1.csv',index=False)
    n_BaIoT_dataset = reset
        
    a = 0
    for name in glob.glob('C:/Users/User/Spyder Warehouse/n-BaIoT/*junk.csv'):
        print("junk folder")
        df = pd.read_csv(name)  
        x = len(df)
        print(x)
        col_1=device_id_adder(x,a)
        df['device_id'] = col_1
        
        col_2=label_adder(x,2)
        df['result'] = col_2
        if (a==0):
            n_BaIoT_dataset = df.copy(deep=True)
        else:
            merge_datasets(df)
        a=a+1
        
    n_BaIoT_dataset.to_csv('n_BaIoT_2.csv',index=False)
    n_BaIoT_dataset = reset
        
    a = 0
    for name in glob.glob('C:/Users/User/Spyder Warehouse/n-BaIoT/*gafgyt.scan.csv'):
        print("gafgyt.scan folder")
        df = pd.read_csv(name)  
        x = len(df)
        print(x)
        col_1=device_id_adder(x,a)
        df['device_id'] = col_1
        
        col_2=label_adder(x,3)
        df['result'] = col_2
        if (a==0):
            n_BaIoT_dataset = df.copy(deep=True)
        else:
            merge_datasets(df)
        a=a+1
        
    n_BaIoT_dataset.to_csv('n_BaIoT_3.csv',index=False)
    n_BaIoT_dataset = reset
    
    a = 0
    for name in glob.glob('C:/Users/User/Spyder Warehouse/n-BaIoT/*tcp.csv'):
        print("tcp folder")
        df = pd.read_csv(name)  
        x = len(df)
        print(x)
        col_1=device_id_adder(x,a)
        df['device_id'] = col_1
        
        col_2=label_adder(x,4)
        df['result'] = col_2
        if (a==0):
            n_BaIoT_dataset = df.copy(deep=True)
        else:
            merge_datasets(df)
        a=a+1
        
    n_BaIoT_dataset.to_csv('n_BaIoT_4.csv',index=False)
    n_BaIoT_dataset = reset
    
    a = 0
    for name in glob.glob('C:/Users/User/Spyder Warehouse/n-BaIoT/*gafgyt.udp.csv'):
        print("gafgyt.udp folder")
        df = pd.read_csv(name)  
        x = len(df)
        print(x)
        col_1=device_id_adder(x,a)
        df['device_id'] = col_1
        
        col_2=label_adder(x,5)
        df['result'] = col_2
        if (a==0):
            n_BaIoT_dataset = df.copy(deep=True)
        else:
            merge_datasets(df)
        a=a+1
        
    n_BaIoT_dataset.to_csv('n_BaIoT_5.csv',index=False)
    n_BaIoT_dataset = reset
    
    a = 0
    for name in glob.glob('C:/Users/User/Spyder Warehouse/n-BaIoT/*ack.csv'):
        print("ack folder")
        df = pd.read_csv(name)  
        x = len(df)
        print(x)
        col_1=device_id_adder(x,a)
        df['device_id'] = col_1
        
        col_2=label_adder(x,6)
        df['result'] = col_2
        if (a==0):
            n_BaIoT_dataset = df.copy(deep=True)
        else:
            merge_datasets(df)
        a=a+1
        
    n_BaIoT_dataset.to_csv('n_BaIoT_6.csv',index=False)
    n_BaIoT_dataset = reset
        
    a = 0
    for name in glob.glob('C:/Users/User/Spyder Warehouse/n-BaIoT/*mirai.scan.csv'):
        print("mirai.scan folder")
        df = pd.read_csv(name)  
        x = len(df)
        print(x)
        col_1=device_id_adder(x,a)
        df['device_id'] = col_1
        
        col_2=label_adder(x,7)
        df['result'] = col_2
        if (a==0):
            n_BaIoT_dataset = df.copy(deep=True)
        else:
            merge_datasets(df)
        a=a+1
        
    n_BaIoT_dataset.to_csv('n_BaIoT_7.csv',index=False)
    n_BaIoT_dataset = reset
    

    a = 0
    for name in glob.glob('C:/Users/User/Spyder Warehouse/n-BaIoT/*syn.csv'):
        print("syn folder")
        df = pd.read_csv(name)  
        x = len(df)
        print(x)
        col_1=device_id_adder(x,a)
        df['device_id'] = col_1
        
        col_2=label_adder(x,8)
        df['result'] = col_2
        if (a==0):
            n_BaIoT_dataset = df.copy(deep=True)
        else:
            merge_datasets(df)
            
    n_BaIoT_dataset.to_csv('n_BaIoT_8.csv',index=False)
    n_BaIoT_dataset = reset
    
        
    
    a = 0
    for name in glob.glob('C:/Users/User/Spyder Warehouse/n-BaIoT/*mirai.udp.csv'):
        print("mirai.udp folder")
        df = pd.read_csv(name)  
        x = len(df)
        print(x)
        col_1=device_id_adder(x,a)
        df['device_id'] = col_1
        
        col_2=label_adder(x,9)
        df['result'] = col_2
        if (a==0):
            n_BaIoT_dataset = df.copy(deep=True)
        else:
            merge_datasets(df)
        a=a+1
    
    n_BaIoT_dataset.to_csv('n_BaIoT_9.csv',index=False)
    n_BaIoT_dataset = reset
        
        
    a = 0
    for name in glob.glob('C:/Users/User/Spyder Warehouse/n-BaIoT/*udpplain.csv'):
        print("udpplain folder")
        df = pd.read_csv(name)  
        x = len(df)
        print(x)
        col_1=device_id_adder(x,a)
        df['device_id'] = col_1
        
        col_2=label_adder(x,10)
        df['result'] = col_2
        if (a==0):
            n_BaIoT_dataset = df.copy(deep=True)
        else:
            merge_datasets(df)
        a=a+1
    
    n_BaIoT_dataset.to_csv('n_BaIoT_10.csv',index=False)

if __name__ == "__main__":
    main()





    
    
