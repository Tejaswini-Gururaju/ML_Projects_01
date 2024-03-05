import sys
import logging
from src.logger import logging


def error_message_details(error,error_details:sys):
    _,_,exc_tb=error_details.exc_info()
    file_name=exc_tb.tb_frame.f_code.co_filename
    error_message = "Error occured in python script  [{0}] line number [{0}]".format(file_name,exc_tb.tb_lineno,str(error))
    return error_message




class CustomException(Exception):
    def __init__(self,error_message,error_detail:sys):
        super().__init__(error_message)
        self.error_message = error_message_details(error_message,error_details=error_detail)

    def __str__(self):
        return self.error_message

