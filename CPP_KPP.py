from llama_index.core.tools import FunctionTool
import statistics

# Function to multiply a list of numbers
def CPP_Calculation(numbers: list) -> float:
    """CPP Calculation"""
    product = 1
    for num in numbers:
        product *= num
    return product

cpp_tool = FunctionTool.from_defaults(fn = CPP_Calculation,
                                           name = "Cpp Formula", 
                                           description = "Critical Process Parameter Formula")


# Function to calculate the average of a list of numbers
def KPP_Calculation(numbers: list) -> float:
    """KPP Calculation"""
    return sum(numbers) / len(numbers) if numbers else 0

kpp_tool = FunctionTool.from_defaults(fn = KPP_Calculation,
                                          name = "KPP Formula",
                                          description = "Key Process Parameter Formula")

# Function to calculate the standard deviation of a list of numbers
def standard_deviation(numbers: list) -> float:
    """Calculate the standard deviation of the numbers in the list"""
    return statistics.stdev(numbers) if len(numbers) > 1 else 0.0

std_dev_tool = FunctionTool.from_defaults(fn = standard_deviation,
                                          name = "Standard Deviation",
                                          description = "This tool is used for Calculation of Standard Deviation.")

