import pandas as pd
import os
import numpy as np


COLUMNS_COVER = ['Parameter_names', 'Parameter_default_values']


class SummaryLogger():
    def __init__(self, all_args, all_default_args, out_path):
        self.out_path = out_path

        self.all_default_args = all_default_args
        self.list_parameter_names = list(all_args.keys())
        self.list_parameter_values = list(all_args.values())

        self.columns = ['Model_name'] + self.list_parameter_names + ['Avg_taw_acc', 'Avg_tag_acc',
                                                                     'Avg_taw_forg',  'Avg_tag_forg',
                                                                     'Avg_perstep_acc_taw','Avg_perstep_acc_tag',
                                                                     'Last_avg_taw_acc', 'Last_avg_tag_acc','Last_avg_perstep_tag_acc'
                                                                     ]
      

   



    def update_summary(self, exp_name, logger):
        list_avg_taw_acc = list(np.around(logger.avg_acc_taw, decimals=3))
        list_avg_tag_acc  = list(np.around(logger.avg_acc_tag, decimals=3))
        list_avg_forg_taw = list(np.around(logger.avg_forg_taw, decimals=3))
        list_avg_forg_tag  = list(np.around(logger.avg_forg_tag , decimals=3))
        
        list_avg_perstep_acc_taw =  list(np.around(logger.avg_perstep_acc_taw, decimals=3))
        list_avg_perstep_acc_tag =  list(np.around(logger.avg_perstep_acc_tag, decimals=3))


        df = pd.DataFrame([[exp_name]+ 
                            self.list_parameter_values+ 
                            ["#".join(str(item) for item in list_avg_taw_acc)]+
                            ["#".join(str(item) for item in list_avg_tag_acc)]+
                            ["#".join(str(item) for item in list_avg_forg_taw)]+
                            ["#".join(str(item) for item in list_avg_forg_tag )]+
                            ["#".join(str(item) for item in list_avg_perstep_acc_taw )]+
                            ["#".join(str(item) for item in list_avg_perstep_acc_tag )]+
                            [list_avg_taw_acc[-1]]+
                            [list_avg_tag_acc[-1]]+
                            [list_avg_perstep_acc_tag[-1]]], columns=self.columns)
        
        df.to_csv(os.path.join(self.out_path, exp_name,  "summary.csv"), index=False)
    

   

 