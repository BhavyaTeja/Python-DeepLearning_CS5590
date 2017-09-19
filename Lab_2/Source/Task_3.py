class Student:
    overall_total = 0

    def __init__(self, name, roll):
        self.full_name = name
        self.rollno = roll
        Student.overall_total += 1

    def count(self):
        print('The number of students are ', Student.overall_total)

    def display(self):
        print('The full name is ', self.full_name)
        print('The roll number is ', self.rollno)

class System:

    def __init__(self, systemType):
        self.TypeOfSystem = systemType


class TransferStudent(Student):

    def __init__(self, name, roll, s):
        super(TransferStudent, self).__init__(name, roll)
        self.TransferredCredits = input(s)

class Grades(TransferStudent):

    def __init__(self, name, roll, s, grade, credits):
        TransferStudent.__init__(self, name, roll, s)
        self.Grades = grade
        self.EnrolledCredits = credits

    def TotalCredits(self):
        self.TotalCreditsEnrolled = self.s + credits






