class Person:
    numPersonObjs = 0

    def __init__(self, name, age):
        self.name = name
        self.age = age
        Person.numPersonObjs += 1

    def __str__(self):
        return self.name + ", " + str(self.age)

    def greet(self):
        print ("hello, I am " + self.__str__())

    @classmethod
    def printNumPersonObjs(cls):
        print ("Number of person Objects is: " + str(Person.numPersonObjs))
