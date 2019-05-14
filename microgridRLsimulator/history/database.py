import pandas as pd

class Database:

    _inputs_ = ['Year', 'Month', 'Day', 'Hour', 'Minutes', 'Seconds', 'IsoDayOfWeek', 'IsoWeekNumber']

    def __init__(self, path_to_csv, grid):
        """
        A Database objects holds the realized data of the microgridRLsimulator in a pandas dataframe.

        The CSV file values are separated by ';' and the first line must contain series names.
        It must contain

        * a 'DateTime' column with values interpretable as python date time objects.
        * a 'Price' column with values interpretable as floats.
        * All the non-flexible quantities (load and generation) described in the microgridRLsimulator configuration

        Some new columns are generated from the DateTime column to indicate e.g. whether
        a datetime corresponds to a day of the week or not.

        :param path_to_csv: Path to csv containing realized data
        :param grid: A Grid object describing the configuration of the microgridRLsimulator
        """
        self._output_ = grid.get_non_flexible_device_names() #+ ['Price'] # Add the price when working on-grid
        self.data_frame = self.read_data(path_to_csv)

    def read_data(self, path):
        """
        Read data and generate new columns based on the DateTime column.

        :param path: Path to the csv data file
        :return: A pandas dataframe
        """
        df = pd.read_csv(path, sep=";|,", parse_dates=True, index_col='DateTime', engine='python')

        df_col_names = list(df.columns.values)

        df['Year'] = df.index.map(lambda x: x.year)
        df['Month'] = df.index.map(lambda x: x.month)
        df['Day'] = df.index.map(lambda x: x.day)
        df['Hour'] = df.index.map(lambda x: x.hour)
        df['Minutes'] = df.index.map(lambda x: x.minute)
        df['Seconds'] = df.index.map(lambda x: x.second)
        df['IsoDayOfWeek'] = df.index.map(lambda x: x.isoweekday())
        df['IsoWeekNumber'] = df.index.map(lambda x: x.isocalendar()[1])

        # Assert required columns are defined
        for tag in self._output_:
            if tag not in df_col_names:
                raise ValueError("Column name %s not defined in %s" % (tag, path))

        return df

    def get_columns(self, column_indexer, time_indexer):
        """

        :param column_indexer: The name of a column
        :param time_indexer: A datetime
        :return: The realized value of the series column_indexer at time time_indexer
        """
        return self.data_frame[column_indexer].get(time_indexer)

    def get_column(self, column_indexer, dt_from, dt_to):
        """

        :param column_indexer: The name of a column
        :param dt_from: A start datetime
        :param dt_to: An end datetime
        :return: A list of values of the column_indexer series between dt_from and dt_to
        """
        return self.data_frame[column_indexer][dt_from:dt_to]

    def get_times(self, time_indexer):
        """

        :param time_indexer: A date time
        :return: A list containing the value of all the series at time time_indexer
        """
        return self.data_frame.loc[time_indexer, self._output_]