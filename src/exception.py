import sys
from src.logger import logging

# we are creating message for the custom exception to be printed in the terminal
def error_message_detail(error,error_detail:sys):
    _,_,exc_tb=error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename if exc_tb else "Unknown Script"
    
    line_number = exc_tb.tb_lineno if exc_tb else "Unknown Line"
    
    error_message = (
        f"\n{'─' * 50}\n"
        f"ERROR: {error}\n"
        f"File:  {file_name}\n"
        f"Line:  {line_number}\n"
        f"{'─' * 50}"
    )

    return error_message

class CustomException(Exception):
    def __init__(self,error_message,error_detail:sys): # here we get error message(e) and error_detail(sys) from the error 
        super().__init__(error_message)
        self.error_message=error_message_detail(error_message,error_detail=error_detail)

    def __str__(self):
        return self.error_message
