class Student:
    overall_total = 0

    def __init__(self):
        self.full_name = input('Enter the name of the student ')
        self.rollno = input('Enter the roll number')
        Student.overall_total += 1

    def count(self):
        print('The number of students enrolled are ', Student.overall_total)

    def display(self):
        print('The full name is ', self.full_name)
        print('The roll number is ', self.rollno)

class System:

    def __init__(self):
        self.TypeOfSystem = input('Enter the system online or inclass: ')

    def display(self):
        print('The system the student enrolled is: ', self.TypeOfSystem)


class TransferStudent(Student):

    def __init__(self, s):
        super(TransferStudent, self).__init__()
        self.TransferredCredits = s

class Grades(TransferStudent):

    def __init__(self, s, grade, credits):
        TransferStudent.__init__(self, s)
        self.Grades = grade
        self.EnrolledCredits = credits

    def TotalCredits(self):
        self.TotalCreditsEnrolled = self.TransferredCredits + credits

    def display(self):
        print('The full name is ', self.full_name)
        print('The roll number is ', self.rollno)
        print('The Transferred credits are: ', self.TransferredCredits)
        print('The total number of credits enrolled: ', self.EnrolledCredits)
        print('The Grade obtained: ', self.Grades)

class Attendance:

    def __init__(self, percentage):
        self.__attendance = percentage
        if self.__attendance < 65:
            print("Student's attendance is low")



Student1 = Student()
Student1.display()

"""
#Student2 = Student("Bhavya", "31")
#Student3 = Student("Prabha", "14")

Student1.display()

#Student4 = TransferStudent("Loke", "45", "5")

Student5 = Grades("Girish", "33", "0", "A+", "15")

Student5.display()

Student5.count()

"""