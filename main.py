import time


class Vector(list):
    def __init__(self, *args):
        self.space = len(args)
        super().__init__(args)

    def __add__(self, other):
        if len(self) == len(other):
            return Vector(*tuple(map(lambda x1, x2: x1 + x2, self, other)))
        else:
            return ValueError("Vectors exist in different dimensions.")

    def __sub__(self, other):
        if len(self) == len(other):
            return Vector(*tuple(map(lambda x1, x2: x1 - x2, self, other)))
        else:
            return ValueError("Vectors exist in different dimensions.")

    def __mul__(self, other):
        if isinstance(other, Vector) and isinstance(self, Vector):
            if len(self) == len(other):
                return sum(tuple(map(lambda x1, x2: x1 * x2, self, other)))
            else:
                return ValueError("Vectors exist in different dimensions.")
        else:
            return Vector(*tuple(map(lambda x1: x1 * other, self)))

    def __abs__(self):
        return pow(sum(tuple(map(lambda x: pow(x, 2), self))), 0.5)

    def projOn(self, direction):
        return (direction) * ((self * direction) / pow(abs(direction), 2))

    def perpTo(self, direction):
        return self - self.projOn(direction)

    def crossWith(self, other):
        if len(self) == 3 and len(other) == 3:
            return Vector(
                (self[1] * other[2] - other[1] * self[2]),
                (-1 * (self[0] * other[2] - other[0] * self[2])),
                (self[0] * other[1] - other[0] * self[1]),
            )
        else:
            return ValueError("Vectors are not in the third dimension.")


# enter rows
# assumes columns, rows, and aug is all correctly sized
# can only take vector as aug
class Matrix(list):
    def __init__(self, *args, aug=[]):
        super().__init__(args)
        self.rows = []
        self.columns = []
        self.aug = Vector(*aug)
        for row in self:
            self.rows.append(Vector(*row))
        for column in range(len(args[0])):
            self.columns.append(
                Vector(*(args[row][column] for row in range(len(args))))
            )

    # __str__ function mainly created by chatgpt
    def __str__(self):
        # Initialize the string that will hold the matrix representation
        matrix_str = "\n"

        # Determine the maximum width of each column
        column_widths = [
            max([len("{:.3f}".format(self.rows[i][j])) for i in range(len(self.rows))])
            for j in range(len(self.rows[0]))
        ]

        # Loop through each row of the matrix
        for i, row in enumerate(self.rows):
            # Format each column of the row
            formatted_row = ["|"]
            for j in range(len(row)):
                formatted_row.append("{:.3f}".format(row[j]).rjust(column_widths[j]))

            # My additions
            formatted_row.append("|")

            if self.aug:
                formatted_row.append(str(self.aug[i]))
                formatted_row.append("|")

            # Append the formatted row to the matrix string
            matrix_str += " ".join(formatted_row) + "\n"

        # Return the final matrix string
        return matrix_str

    def __mul__(self, other):
        new = []

        # vector matrix multipication
        if isinstance(other, Vector):
            temp = []
            try:
                if other.space != len(self.columns):
                    raise TypeError("Vector space does not match matrix space")
                for i, col in enumerate(self.columns):
                    temp.append(Vector(*col) * other[i])
                temp = Matrix(*temp).T()
                for i, row in enumerate(temp):
                    new.append(sum(row))
                return Vector(*new)
            except Exception as e:
                print(e)
            
        # matrix vector multiplication
        elif isinstance(other, Matrix):
            if len(self.rows) == len(other.columns):
                for row in range(len(self.rows)):
                    new.append(
                        list(
                            self.rows[row] * other.columns[col]
                            for col in range(len(other.columns))
                        )
                    )
            return Matrix(*new)

    def cof(self, rowI, colI):
        new = []
        for i, row in enumerate(self.rows):
            if i != rowI:
                r = []
                for x, num in enumerate(row):
                    if x != colI:
                        r.append(num)
                new.append(r)
        return Matrix(*new)

    # shows all the determinats
    def detS(self):
        if len(self.rows) == 2:
            return (self[0][0] * self[1][1]) - (self[1][0] * self[0][1])
        else:
            result = []
            for col, num in enumerate(self.rows[0]):
                print(f"{num} * determinant of: {self.cof(0,col)}")
                val = self.cof(0, col).detS()
                if col % 2:
                    val = val * -1
                print(f"The sum was: {num*val}")

                result.append(num * val)

            return sum(result)

    def det(self):
        if len(self.rows) == 2:
            return (self[0][0] * self[1][1]) - (self[1][0] * self[0][1])
        else:
            result = []
            for col, num in enumerate(self.rows[0]):
                val = self.cof(0, col).det()
                if col % 2:
                    val = val * -1
                result.append(num * val)
            return sum(result)

    def rref(self):
        new = [*self.rows]
        newAug = [*self.aug]

        # finds smallest dimension, may make a funtion to do this
        for size in range(
            self.rows[0].space
            if self.rows[0].space < self.columns[0].space
            else self.columns[0].space
        ):
            # the if else is to avoid dividing by zero incase to rows are Identical

            leadingOneFactor = 1 / new[size][size]
            new[size] = (
                new[size] * (leadingOneFactor) if leadingOneFactor else new[size]
            )
            newAug[size] = (
                newAug[size] * (leadingOneFactor) if leadingOneFactor else newAug[size]
            )

            for rowIndex, row in enumerate(new):
                if rowIndex != size:
                    zeroingFactor = row[size]
                    new[rowIndex] = row - (new[size] * zeroingFactor)
                    newAug[rowIndex] = newAug[rowIndex] - (newAug[size] * zeroingFactor)

        return Matrix(*new, aug=newAug)
    
    def T(self):
        rows = self.columns
        return Matrix(*rows)

