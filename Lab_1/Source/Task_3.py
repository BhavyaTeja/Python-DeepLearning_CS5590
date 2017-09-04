# Program to find those numbers which are divisible by 5 and multiple of 2, between 700 and 1700

# Creating a list to take all the numbers

numbers = []

# Adding the numbers which are divisible by 5 and multiple of 2 into the list

for i in range(700, 1701):
    if i%5 == 0 and i%2 == 0:                                  # Checking the condition
        numbers.append(str(i))                                 # Appending the numbers that satisfy the condition to the list

# Printing the numbers that are in the list

print("The numbers between 700 and 1700 that are divisible by 5 and are multiple of 2 are: \n")
print(numbers)

