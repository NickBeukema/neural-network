import csv
import pdb
import functions

yes_no = {
    "Yes": 1,
    "No": -1
}

def game_data_format():

    translations = {
        "Health": {
            "Poor": [0,0,1],
            "Fair": [0,1,0],
            "Good": [-1,-1,-1]
        },
        "Armor": yes_no,
        "Weapon": yes_no,
        "Enemies": "continuous",
        "Action": {
            "Wander": [1,0,0,0],
            "Hide":   [0,1,0,0],
            "Attack": [0,0,1,0],
            "Run":    [0,0,0,1]
        }
    }

    columns = ["Health", "Armor", "Weapon", "Enemies", "Action"]

    return {
        "translations": translations,
        "columns": columns
    }

def iris_data_format():

    translations = {
        "Sepal Length": "continuous",
        "Sepal Width": "continuous",
        "Petal Length": "continuous",
        "Petal Width": "continuous",
        "Iris Type": {
            "Setosa": [1,0,0],
            "Versicolor": [0,1,0],
            "Virginica": [0,0,1]
        }
    }

    columns = ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width", "Iris Type"]

    return {
        "translations": translations,
        "columns": columns
    }

def check_digit(value):

    if value.isdigit():
        return True

    try:
        x = float(value)
    except ValueError:
        return False



    return True

def standardize(reader, class_count):
    rows = []

    for row in reader:
        rows.append(row)

    attributes = len(rows[0]) - class_count

    for i in range(attributes):
        if check_digit(rows[0][i]):

            values = []

            for row in rows:
                values.append(row[i])

            mean = functions.mean(values)
            stddev = functions.standard_deviation(values, mean)

            for row in rows:
                row[i] = (float(row[i]) - mean) / stddev

    return rows

def parse(filename, translations):
    data = []

    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')

        rows = standardize(reader, 1)
        for row in rows:
            for i in range(len(row)):
                translation = translations["translations"][translations["columns"][i]]
                if translation is not "continuous":
                    row[i] = translation[row[i]]
                else:
                    row[i] = float(row[i])

            new_row = []

            for i in row:
                if isinstance(i, list):
                    for j in i:
                        new_row.append(j)
                else:
                    new_row.append(i)

            data.append(new_row)

    return data


# data = parse('game-data.csv', game_data_format())
# pdb.set_trace()
