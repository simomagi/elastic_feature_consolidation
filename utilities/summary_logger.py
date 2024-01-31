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
        
        # Per_step_taw_acc: per step accuracy task aware after each task

        # Last_per_step_taw_acc: Last per step task aware accuracy 

        # Per_step_tag_acc: per step accuracy task agnostic after each task (useful for the plots)
        # Last_per_step_tag_acc: Last per step task agnostic accuracy (Equation 16 left in the main paper)
        # Average_inc_accuracy: average incremental accuracy of ICARL (task agnostic) (Equation 16 right in the main paper)
      
        self.columns = ['Model_name'] + self.list_parameter_names + ['Per_step_taw_acc', 
                                                                     'Last_per_step_taw_acc',
                                                                     'Per_step_tag_acc',
                                                                     'Last_per_step_tag_acc',
                                                                     'Average_inc_acc']
      
      
    def update_summary(self, exp_name, logger):
        list_perstep_acc_taw  =  list(np.around(logger.list_perstep_acc_taw, decimals=3))
        list_perstep_acc_tag =  list(np.around(logger.list_perstep_acc_tag, decimals=3))
        average_incremental_acc = np.around(np.mean(list_perstep_acc_tag), decimals=3)

        df = pd.DataFrame([[exp_name]+ 
                            self.list_parameter_values+
                            ["#".join(str(item) for item in list_perstep_acc_taw)]+
                            [list_perstep_acc_taw[-1]]+
                            ["#".join(str(item) for item in list_perstep_acc_tag)]+
                            [list_perstep_acc_tag[-1]]]+
                            [average_incremental_acc], columns=self.columns)
        
        df.to_csv(os.path.join(self.out_path, exp_name,  "summary.csv"), index=False)
    

   

 