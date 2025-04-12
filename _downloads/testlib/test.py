def testfunc(paramA, paramB, paramExtra):
    '''
    test independent function
    Parameters
    ----------
    paramA : boolean
        a number, paramA must be True when paramB is less than 20
    paramB : int
        a number for testing
    '''
    if paramB < 20:
        if paramA is not None:
            raise ValueError("error message")

def testfunc2(paramX, paramY):
    '''
    test independent function
    Parameters
    ----------
    paramX : boolean
        paramX must be True
    paramY : int
        paramY must be less than 10
    '''
    print(paramX)
    print(paramY)

class TestClass:
    '''
    test class and member functions
    Parameters
    ----------
    paramC : default=None
        if paramC is None, paramD must be None
    paramD : default=None
        paramD
    paramE : str
        paramE can only be 'baz' when paramF='foo'
    paramF : str
        paramF can be set to 'foo' or 'bar'

    Attributes
    ----------
    attrA : boolean
        attrA
    attrB : int
        AttrB
    '''

    def __init__(self, attrA, attrB):
        self.attrA = attrA
        self.attrB = attrB

    def test_memfunc1(self, paramC, paramD):
        if paramC is None:
            if paramD is not None:
                raise ValueError("error message")

    def test_memfunc2(self, paramE, paramF):
        if paramF == 'foo':
            if paramE != 'baz':
                raise ValueError("error message")
            else:
                print("Pass, paramE is baz")