#ifndef MatrixC_MatrixH
#define MatrixC_MatrixH

#include <vector>
#include <stdexcept>
#include <cmath>

namespace MatrixC
{
    template<typename T>
    class Matrix
    {
    private:
        std::vector<std::vector<T>> _table;
        std::size_t _rowCount;
        std::size_t _columnCount;

    public:
        Matrix(std::size_t rowCount, std::size_t columnCount)
        {
            if (rowCount == 0 || columnCount == 0)
                throw std::invalid_argument("rowCount and columnCount must be positive integer");

            _rowCount = rowCount;
            _columnCount = columnCount;

            _table.resize(_rowCount);
            for (std::size_t i = 0; i < _rowCount; ++i)
                _table[i].resize(_columnCount);
        }

        Matrix(const Matrix& matrix) noexcept
        {
            _rowCount = matrix.RowCount();
            _columnCount = matrix.ColumnCount();

            _table.resize(_rowCount);
            for (std::size_t i = 0; i < _rowCount; ++i)
                _table[i] = matrix.GetRow(i);
        }

        Matrix(Matrix&& matrix) noexcept
        {
            _rowCount = matrix.RowCount();
            _columnCount = matrix.ColumnCount();

            _table.resize(_rowCount);
            for (std::size_t i = 0; i < _rowCount; ++i)
                _table[i] = std::move(matrix[i]);
        }

        ~Matrix() noexcept = default;

        Matrix& operator=(const Matrix& matrix) noexcept
        {
            _rowCount = matrix.RowCount();
            _columnCount = matrix.ColumnCount();

            _table.resize(_rowCount);
            for (std::size_t i = 0; i < _rowCount; ++i)
                _table[i] = matrix.GetRow(i);

            return *this;
        }

        Matrix& operator=(Matrix&& matrix) noexcept
        {
            _rowCount = matrix.RowCount();
            _columnCount = matrix.ColumnCount();

            _table.resize(_rowCount);
            for (std::size_t i = 0; i < _rowCount; ++i)
                _table[i] = std::move(matrix[i]);

            return *this;
        }

        bool operator==(const Matrix& matrix) const noexcept
        {
            if (_rowCount != matrix.RowCount() || _columnCount != matrix.ColumnCount())
                return false;

            for (std::size_t i = 0; i < _rowCount; ++i)
                for (std::size_t j = 0; j < _columnCount; ++j)
                    if (_table[i][j] != matrix.GetElement(i, j)) return false;

            return true;
        }

        bool operator!=(const Matrix& matrix) const noexcept
        {
            if (_rowCount != matrix.RowCount() || _columnCount != matrix.ColumnCount())
                return true;

            for (std::size_t i = 0; i < _rowCount; ++i)
                for (std::size_t j = 0; j < _columnCount; ++j)
                    if (_table[i][j] == matrix.GetElement(i, j)) return false;

            return true;
        }

        Matrix operator+(const Matrix& matrix) const
        {
            if (_rowCount != matrix.RowCount() || _columnCount != matrix.ColumnCount())
                throw std::invalid_argument("Size of matrix must be equal");

            Matrix result(_rowCount, _columnCount);
            for (std::size_t i = 0; i < _rowCount; ++i)
                for (std::size_t j = 0; j < _columnCount; ++j)
                    result[i][j] = _table[i][j] + matrix.GetElement(i, j);

            return std::move(result);
        }

        Matrix operator-(const Matrix& matrix) const
        {
            if (_rowCount != matrix.RowCount() || _columnCount != matrix.ColumnCount())
                throw std::invalid_argument("Size of matrix must be equal");

            Matrix result(_rowCount, _columnCount);
            for (std::size_t i = 0; i < _rowCount; ++i)
                for (std::size_t j = 0; j < _columnCount; ++j)
                    result[i][j] = _table[i][j] - matrix.GetElement(i, j);

            return std::move(result);
        }

        Matrix operator*(const Matrix& matrix) const
        {
            if (_columnCount != matrix.RowCount())
                throw std::invalid_argument("Column size of left matrix must be equal to row size of right matrix");

            Matrix result(_rowCount, matrix.ColumnCount());

            for (std::size_t i = 0; i < _rowCount; ++i)
                for (std::size_t j = 0; j < matrix.ColumnCount(); ++j)
                {
                    T c{};
                    for (std::size_t k = 0; k < _columnCount; ++k)
                        c += _table[i][k] * matrix.GetElement(k, j);

                    result[i][j] = c;
                }

            return std::move(result);
        }

        Matrix operator*(const T& alpha) const noexcept
        {
            Matrix result(_rowCount, _columnCount);
            for (std::size_t i = 0; i < _rowCount; ++i)
                for (std::size_t j = 0; j < _columnCount; ++j)
                    result[i][j] = _table[i][j] * alpha;

            return std::move(result);
        }

        Matrix operator*(T&& alpha) const noexcept
        {
            Matrix result(_rowCount, _columnCount);
            for (std::size_t i = 0; i < _rowCount; ++i)
                for (std::size_t j = 0; j < _columnCount; ++j)
                    result[i][j] = _table[i][j] * alpha;

            return std::move(result);
        }

        std::vector<T>& operator[](std::size_t rowNumber)
        {
            return _table[rowNumber];
        }

        Matrix Transpose() const noexcept
        {
            Matrix result(_columnCount, _rowCount);
            for (std::size_t i = 0; i < _rowCount; ++i)
                for (std::size_t j = 0; j < _columnCount; ++j)
                    result[j][i] = _table[i][j];

            return std::move(result);
        }

        void EraseRow(std::size_t rowNumber)
        {
            if (_rowCount == 1)
                throw std::invalid_argument("You cannot erase a row from a matrix that has only one row");

            _table.erase(_table.cbegin() + rowNumber);
            --_rowCount;
        }

        void EraseColumn(std::size_t columnNumber)
        {
            if (_columnCount == 1)
                throw std::invalid_argument("You cannot erase a column from a matrix that has only one column");

            for (std::size_t i = 0; i < _rowCount; ++i)
                _table[i].erase(_table[i].cbegin() + columnNumber);
            --_columnCount;
        }

        T AlgebraicComplement(std::size_t i, std::size_t j) const
        {
            Matrix minorMatrix(*this);
            minorMatrix.EraseRow(i);
            minorMatrix.EraseColumn(j);
            return std::pow(-1, i + j + 2) * minorMatrix.Det();
        }

        T Det() const
        {
            if (_rowCount != _columnCount)
                throw std::range_error("The matrix must be square");

            if (_rowCount == 1)
                return _table[0][0];

            T result{};
            for (std::size_t i = 0; i < _rowCount; ++i)
                result += _table[i][0] * AlgebraicComplement(i, 0);

            return std::move(result);
        }

        Matrix Inverse() const
        {
            const T det = Det();
            if (!det)
                throw std::logic_error("Det must not be zero");

            Matrix inverseMatrix(_rowCount, _columnCount);
            for (std::size_t i = 0; i < _rowCount; ++i)
                for (std::size_t j = 0; j < _columnCount; ++j)
                {
                    T a = AlgebraicComplement(i, j);
                    inverseMatrix[j][i] = a / det;
                }

            return std::move(inverseMatrix);
        }

        void SwapRow(std::size_t row1Number, std::size_t row2Number)
        {
            std::swap(_table[row1Number], _table[row2Number]);
        }

        void SwapColumn(std::size_t column1Number, std::size_t column2Number)
        {
            for (std::size_t i = 0; i < _rowCount; ++i)
                std::swap(_table[i][column1Number], _table[i][column2Number]);
        }

        std::size_t Rank() const noexcept
        {
            std::size_t rank = _columnCount;
            Matrix copy(*this);

            for (std::size_t rowIndex = 0; rowIndex < rank; ++rowIndex)
            {
                if (copy[rowIndex][rowIndex])
                {
                    for (std::size_t columnIndex = 0; columnIndex < _rowCount; ++columnIndex)
                        if (columnIndex != rowIndex)
                        {
                            T multiplier = copy[columnIndex][rowIndex] / copy[rowIndex][rowIndex];
                            for (std::size_t i = 0; i < rank; ++i)
                                copy[columnIndex][i] -= multiplier * copy[rowIndex][i];
                        }
                }
                else
                {
                    bool reduce = true;

                    for (std::size_t i = rowIndex + 1; i < _rowCount; ++i)
                        if (copy[i][rowIndex])
                        {
                            copy.SwapRow(rowIndex, i);
                            reduce = false;
                            break;
                        }

                    if (reduce)
                    {
                        --rank;
                        for (int i = 0; i < _rowCount; ++i)
                            copy[i][rowIndex] = copy[i][rank];
                    }
                    --rowIndex;
                }
            }

            return rank;
        }

        inline std::size_t RowCount() const noexcept
        {
            return _rowCount;
        }

        inline std::size_t ColumnCount() const noexcept
        {
            return _columnCount;
        }

        std::vector<T> GetRow(std::size_t rowNumber) const
        {
            return _table[rowNumber];
        }

        void SetRow(const std::vector<T>& newRow, std::size_t rowNumber)
        {
            if (newRow.size() != _columnCount)
                throw std::invalid_argument("Incorrect newRow size");

            _table[rowNumber] = newRow;
        }

        void SetRow(std::vector<T>&& newRow, std::size_t rowNumber)
        {
            if (newRow.size() != _columnCount)
                throw std::invalid_argument("Incorrect newRow size");

            _table[rowNumber] = newRow;
        }

        std::vector<T> GetColumn(std::size_t columnNumber) const
        {
            std::vector<T> column(_rowCount);
            for (std::size_t i = 0; i < _rowCount; ++i)
                column[i] = _table[i][columnNumber];
            return column;
        }

        void SetColumn(const std::vector<T>& newColumn, std::size_t columnNumber)
        {
            if (newColumn.size() != _rowCount)
                throw std::invalid_argument("Incorrect newColumn size");

            for (std::size_t i = 0; i < _rowCount; ++i)
                _table[i][columnNumber] = newColumn[i];
        }

        void SetColumn(std::vector<T>&& newColumn, std::size_t columnNumber)
        {
            if (newColumn.size() != _rowCount)
                throw std::invalid_argument("Incorrect newColumn size");

            for (std::size_t i = 0; i < _rowCount; ++i)
                _table[i][columnNumber] = std::move(newColumn[i]);
        }

        inline T GetElement(std::size_t i, std::size_t j) const
        {
            return _table[i][j];
        }

        inline void SetElement(const T& newValue, std::size_t i, std::size_t j)
        {
            _table[i][j] = newValue;
        }

        inline void SetElement(T&& newValue, std::size_t i, std::size_t j)
        {
            _table[i][j] = newValue;
        }

        void Resize(std::size_t rowCount, std::size_t columnCount)
        {
            if (rowCount == 0 || columnCount == 0)
                throw std::invalid_argument("Matrix must not be empty");

            for (std::size_t i = 0; i < _rowCount; ++i)
                _table[i].resize(columnCount);
            _table.resize(rowCount);

            _rowCount = rowCount;
            _columnCount = columnCount;
        }

        std::string ToString() const noexcept
        {
            std::string matrixString;

            for (std::size_t i = 0; i < _rowCount; ++i)
            {
                for (std::size_t j = 0; j < _columnCount; ++j)
                    matrixString += std::to_string(_table[i][j]) + "\t";
                matrixString += "\n";
            }

            return std::move(matrixString);
        }
    };
}

typedef MatrixC::Matrix<std::double_t> Matrix;
typedef MatrixC::Matrix<std::double_t> MatrixD;
typedef MatrixC::Matrix<std::float_t> MatrixF;
typedef MatrixC::Matrix<std::int64_t> MatrixL;
typedef MatrixC::Matrix<std::int32_t> MatrixI;
typedef MatrixC::Matrix<std::int16_t> MatrixS;
typedef MatrixC::Matrix<std::int8_t> MatrixB;

#define ZERO_MATRIX(rowCount, columnCount) \
return MatrixD(rowCount, columnCount);

#define E_MATRIX(rowCount, columnCount) \
MatrixD matrix(rowCount, columnCount); \
for (std::size_t i = 0, j = 0; i < rowCount && j < columnCount; ++i, ++j) \
    matrix[i][j] = 1; \
return matrix;

#endif