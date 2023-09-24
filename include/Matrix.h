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
                throw std::invalid_argument("");

            _table.resize(rowCount);
            for (std::size_t i = 0; i < rowCount; ++i)
                _table[i].resize(columnCount);

            _rowCount = rowCount;
            _columnCount = columnCount;
        }

        Matrix(const Matrix& matrix)
        {
            _table.resize(matrix.RowCount());
            for (std::size_t i = 0; i < matrix.RowCount(); ++i)
            {
                _table[i].resize(matrix.ColumnCount());
                for (std::size_t j = 0; j < matrix.ColumnCount(); ++j)
                    _table[i][j] = matrix.GetElement(i, j);
            }

            _rowCount = matrix.RowCount();
            _columnCount = matrix.ColumnCount();
        }

        ~Matrix() = default;

        Matrix& operator=(const Matrix& matrix)
        {
            _table.resize(matrix.RowCount());
            for (std::size_t i = 0; i < matrix.RowCount(); ++i)
            {
                _table[i].resize(matrix.ColumnCount());
                for (std::size_t j = 0; j < matrix.ColumnCount(); ++j)
                    _table[i][j] = matrix.GetElement(i, j);
            }

            _rowCount = matrix.RowCount();
            _columnCount = matrix.ColumnCount();

            return *this;
        }

        bool operator==(const Matrix& matrix) const
        {
            if (_rowCount != matrix.RowCount() || _columnCount != matrix.ColumnCount())
                return false;

            for (std::size_t i = 0; i < _rowCount; ++i)
                for (std::size_t j = 0; j < _columnCount; ++j)
                    if (_table[i][j] != matrix.GetElement(i, j)) return false;

            return true;
        }

        bool operator!=(const Matrix& matrix) const
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
                throw std::invalid_argument("");

            Matrix result(_rowCount, _columnCount);
            for (std::size_t i = 0; i < _rowCount; ++i)
                for (std::size_t j = 0; j < _columnCount; ++j)
                    result[i][j] = _table[i][j] + matrix.GetElement(i, j);

            return result;
        }

        Matrix operator-(const Matrix& matrix) const
        {
            if (_rowCount != matrix.RowCount() || _columnCount != matrix.ColumnCount())
                throw std::invalid_argument("");

            Matrix result(_rowCount, _columnCount);
            for (std::size_t i = 0; i < _rowCount; ++i)
                for (std::size_t j = 0; j < _columnCount; ++j)
                    result[i][j] = _table[i][j] - matrix.GetElement(i, j);

            return result;
        }

        Matrix operator*(const Matrix& matrix) const
        {
            if (_columnCount != matrix.RowCount())
                throw std::invalid_argument("");

            Matrix result(_rowCount, matrix.ColumnCount());

            for (std::size_t i = 0; i < _rowCount; ++i)
                for (std::size_t j = 0; j < matrix.ColumnCount(); ++j)
                {
                    T c{};
                    for (std::size_t k = 0; k < _columnCount; ++k)
                        c += _table[i][k] * matrix.GetElement(k, j);

                    result[i][j] = c;
                }

            return result;
        }

        Matrix operator*(const T&& alpha) const
        {
            Matrix result(_rowCount, _columnCount);
            for (std::size_t i = 0; i < _rowCount; ++i)
                for (std::size_t j = 0; j < _columnCount; ++j)
                    result[i][j] = _table[i][j] * alpha;

            return result;
        }

        std::vector<T>& operator[](std::size_t rowNumber)
        {
            return _table[rowNumber];
        }

        Matrix& EraseRow(std::size_t rowNumber)
        {
            if (_rowCount == 1)
                throw std::invalid_argument("");

            _table.erase(_table.cbegin() + rowNumber);
            _rowCount -= 1;

            return *this;
        }

        Matrix& EraseColumn(std::size_t columnNumber)
        {
            if (_columnCount == 1)
                throw std::invalid_argument("");

            for (std::size_t i = 0; i < _rowCount; ++i)
                _table[i].erase(_table[i].cbegin() + columnNumber);
            _columnCount -= 1;

            return *this;
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
                throw std::range_error("");

            if (_rowCount == 1)
                return _table[0][0];

            T result{};
            for (std::size_t i = 0; i < _rowCount; ++i)
                result += _table[i][0] * AlgebraicComplement(i, 0);

            return result;
        }

        Matrix Inverse() const
        {
            const T det = Det();
            if (!det)
                throw std::logic_error("");

            Matrix inverseMatrix(_rowCount, _columnCount);
            for (std::size_t i = 0; i < _rowCount; ++i)
                for (std::size_t j = 0; j < _columnCount; ++j)
                {
                    T a = AlgebraicComplement(i, j);
                    inverseMatrix[j, i] = a / det;
                }

            return inverseMatrix;
        }

        std::size_t Rang() const
        {
            for (std::size_t rang = std::min(_rowCount, _columnCount); rang > 0; --rang)
            {
                Matrix matrix(rang, rang);
                for (std::size_t i = 0; i < rang; ++i)
                {
                    matrix[i] = _table[i];
                    matrix[i].resize(rang);
                }

                if (matrix.Det()) return rang;
            }

            return 0;
        }

        std::size_t RowCount() const
        {
            return _rowCount;
        }

        std::size_t ColumnCount() const
        {
            return _columnCount;
        }

        std::vector<T> GetRow(std::size_t rowNumber) const
        {
            return _table[rowNumber];
        }

        Matrix& SetRow(const std::vector<T>& newRow, std::size_t rowNumber)
        {
            if (newRow.size() != _columnCount)
                throw std::invalid_argument("");

            _table[rowNumber] = newRow;

            return *this;
        }

        std::vector<T> GetColumn(std::size_t columnNumber) const
        {
            std::vector<T> column(_rowCount);
            for (std::size_t i = 0; i < _rowCount; ++i)
                column[i] = _table[i][columnNumber];
            return column;
        }

        Matrix& SetColumn(const std::vector<T>& newColumn, std::size_t columnNumber)
        {
            if (newColumn.size() != _rowCount)
                throw std::invalid_argument("");

            for (std::size_t i = 0; i < _rowCount; ++i)
                _table[i][columnNumber] = newColumn[i];

            return *this;
        }

        T GetElement(std::size_t i, std::size_t j) const
        {
            return _table[i][j];
        }

        Matrix& SetElement(const T&& newValue, std::size_t i, std::size_t j)
        {
            _table[i][j] = newValue;
            return *this;
        }

        Matrix& Resize(std::size_t rowCount, std::size_t columnCount)
        {
            if (rowCount == 0 || columnCount == 0)
                throw std::invalid_argument("");

            for (std::size_t i = 0; i < _rowCount; ++i)
                _table[i].resize(columnCount);
            _table.resize(rowCount);

            _rowCount = rowCount;
            _columnCount = columnCount;

            return *this;
        }

        std::string ToString() const
        {
            std::string matrixString;

            for (std::size_t i = 0; i < _rowCount; ++i)
            {
                for (std::size_t j = 0; j < _columnCount; ++j)
                    matrixString += std::to_string(_table[i][j]) + "\t";
                matrixString += "\n";
            }

            return matrixString;
        }
    };
}

typedef MatrixC::Matrix<double> Matrix;
typedef MatrixC::Matrix<double> MatrixD;
typedef MatrixC::Matrix<float> MatrixF;
typedef MatrixC::Matrix<long long> MatrixL;
typedef MatrixC::Matrix<int> MatrixI;
typedef MatrixC::Matrix<short> MatrixS;

#define ZERO_MATRIX(rowCount, columnCount) \
return MatrixD(rowCount, columnCount);

#define E_MATRIX(rowCount, columnCount) \
MatrixD matrix(rowCount, columnCount); \
for (std::size_t i = 0, j = 0; i < rowCount && j < columnCount; ++i, ++j) \
    matrix[i][j] = 1; \
return matrix;

#endif