"""Python lin alg module"""


class Vector(list):
    """Define a vector"""

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

    def proj_on(self, direction):
        """Find the projection of one vector onto another."""
        return (direction) * ((self * direction) / pow(abs(direction), 2))

    def perp_to(self, direction):
        """Find the perpendicular of one vector to another."""
        return self - self.proj_on(direction)

    def cross_with(self, other):
        """Get the cross product of two vectors. Both must be in R3"""
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
    """
    Enter rows, leave out encapsulating square brackets e.g. Matrix([1,1], [1,1]) 
    not Matrix([[1,1], [1,1]]).
    """

    def __init__(self, *args, aug: "Matrix" = None):
        super().__init__(args)
        self.rows = []
        self.columns = []
        self.aug = aug
        if aug:
            self.aug = aug
        if args:
            for row in self:
                self.rows.append(Vector(*row))
            for column in range(len(args[0])):
                self.columns.append(
                    Vector(*(args[row][column] for row in range(len(args))))
                )

    def __getitem__(self, index) -> Vector | float:
        return self.rows[index]

    def __setitem__(self, index, value) -> None:
        if isinstance(index, tuple):
            row, col = index
            self[row][col] = value
            self.columns[col][row] = value
            self.rows[row][col] = value
        else:
            super().__setitem__(index, value)
            self.rows[index] = Vector(*value)
            if len(self.columns) > 0:
                for row in range(len(self)):
                    self.columns[row][index] = self.rows[index][row]

    def __repr__(self):
        rows = []
        for row in self.rows:
            row_str = " ".join([str(elem) for elem in row])
            rows.append(row_str)
        return "\n".join(rows)

    def __str__(self) -> str:
        # Initialize the string that will hold the matrix representation
        matrix_str = "\n"

        # Determine the maximum width of each column
        column_widths = [
            max([len("{:.3f}".format(self.rows[i][j]))
                for i in range(len(self.rows))])
            for j in range(len(self.rows[0]))
        ]

        if isinstance(self.aug, Vector):
            aug_width = [
                max([len("{:.3f}".format(self.aug[i]))
                    for i in range(len(self.aug))])
            ]

        if isinstance(self.aug, Matrix):
            aug_widths = [
                max([len("{:.3f}".format(self.aug[i][j]))
                    for i in range(len(self.aug))])
                for j in range(len(self.aug[0]))
            ]

        # Loop through each row of the matrix
        for i, row in enumerate(self.rows):
            # Format each column of the row
            formatted_row = ["|"]
            for j, _ in enumerate(row):
                formatted_row.append("{:.3f}".format(
                    row[j]).rjust(column_widths[j]))

            # My additions
            if isinstance(self.aug, Vector):

                formatted_row.append("|")
                # Right-align the augmented column
                formatted_row.append("{:.3f}".format(
                    (self.aug[i])).rjust(max(aug_width)))
                formatted_row.append("|")

            elif isinstance(self.aug, Matrix):
                print("here")
                formatted_row.append("|")
                for j, _ in enumerate(row):
                    formatted_row.append("{:.3f}".format(
                        self.aug[i][j]).rjust(aug_widths[j]))
                formatted_row.append("|")
            else:
                formatted_row.append("|")

            # Append the formatted row to the matrix string
            matrix_str += " ".join(formatted_row) + "\n"

        return matrix_str

    def __mul__(self, other):
        new = []

        # vector matrix multipication
        if isinstance(other, Vector):
            temp = []
            try:
                if other.space != len(self.columns):
                    raise ValueError(
                        "Vector space does not match matrix space")
                for i, col in enumerate(self.columns):
                    temp.append(Vector(*col) * other[i])
                temp = Matrix(*temp).T()
                for i, row in enumerate(temp):
                    new.append(sum(row))
                return Vector(*new)
            except ValueError as exception:
                print(exception)

        # matrix vector multiplication
        elif isinstance(other, Matrix):
            if len(self.columns) == len(other.rows):
                for col in other.columns:
                    new.append(self * col)
            return Matrix(*new).T()

    def cof(self, row_i, col_i) -> "Matrix":
        """Get co factor matrix."""
        new = []
        for i, row in enumerate(self.rows):
            if i != row_i:
                r = []
                for x, num in enumerate(row):
                    if x != col_i:
                        r.append(num)
                new.append(r)
        return Matrix(*new)

    # shows all the determinats
    def det_show(self) -> float:
        "Get determinant. Prints all steps"
        if len(self.rows) == 2:
            return (self[0][0] * self[1][1]) - (self[1][0] * self[0][1])
        else:
            result = []
            for col, num in enumerate(self.rows[0]):
                print(f"{num} * determinant of: {self.cof(0,col)}")
                val = self.cof(0, col).det_show()
                if col % 2:
                    val = val * -1
                print(f"The sum was: {num*val}")

                result.append(num * val)

            return sum(result)

    def det(self) -> float:
        """Gets determinant."""
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

    def rref(self) -> "Matrix":
        """Gets reduced row echlon form of matrix"""
        new = [*self.rows]
        new_aug = self.aug

        # case where the matrix is 1x1
        if len(new) == 1 and len(new[0]) == 1:
            if new[0][0] != 0:
                new_aug[0, 0] = new_aug[0][0] / new[0][0]
                new[0][0] = 1
            return Matrix(*new, aug=new_aug)

        # finds smallest dimension, may make a funtion to do this
        for row_index in range(
            self.rows[0].space
            if self.rows[0].space < self.columns[0].space
            else self.columns[0].space
        ):
            # rotates rows until the val at M(nn) is not 0 won't rotate rows above ROW n
            rotations = 0
            column_normalizable = True
            while new[row_index][row_index] == 0:
                rotations += 1

                # apply to matrix
                last_row = new.pop(-1)
                new.insert(row_index, last_row)

                # apply to augmented matrix if there is one
                if new_aug:
                    last_row_aug = new_aug.pop(-1)
                    new_aug.insert(row_index, last_row_aug)

                # breaks out of loop if max rotations have been done
                if rotations == len(self.rows) - row_index:
                    column_normalizable = False
                    break

            # if a column has all 0s it will skip to the next column
            if not column_normalizable:
                continue

            leading_one_factor = 1 / new[row_index][row_index]

            new[row_index] = new[row_index] * leading_one_factor

            if new_aug:
                new_aug[row_index] = new_aug[row_index] * leading_one_factor

            # subtracts row with leading 1 from all other rows to make everything else in the column 0
            for i, row in enumerate(new):
                if row_index != i:
                    zeroing_factor = row[row_index]
                    new[i] = row - (new[row_index] * zeroing_factor)

                    if new_aug:
                        new_aug[i] = new_aug[i] - \
                            (new_aug[row_index] * zeroing_factor)

        return Matrix(*new, aug=new_aug)

    def inverse(self) -> "Matrix":
        "Returns either the appropriate (left, right, square) inverse or false if matrix is not invertable"

        # checks the matrix is square
        if len(self.rows) == len(self.columns):
            # creates identity matrix
            new_aug = Matrix(
                *[
                    [1 if col == row else 0 for col in range(len(self.rows))]
                    for row in range(len(self.rows))
                ]
            )

            inverse = Matrix(*self.rows, aug=new_aug).rref().aug

            return inverse

        # checks to see if right inverse is appropriate
        if len(self.rows) < len(self.columns):
            transpose = self.T()

            # multiply the matrix by its transpose to make it square
            square = self * transpose

            square_inverse = square.inverse()
            # reverses the "squaring" action done above
            inverse = transpose * square_inverse

            return inverse

        # checks to see if left inverse is appropriate
        if len(self.rows) > len(self.columns):
            transpose = self.T()

            # multiply the matrix by its transpose to make it square
            square = transpose * self

            # reverses the "squaring" action done above
            inverse = square.inverse() * transpose

            return inverse

        return False

    def rank(self) -> int:
        """Returns rank of matrix."""
        rref = self.rref()
        rank = 1

        min_size = min(len(rref.columns), len(rref.rows))

        for val in range(min_size):
            if val == 1:
                rank += 1

        return rank

    def T(self):
        """Get transposed matrix."""
        rows = self.columns
        return Matrix(*rows)
