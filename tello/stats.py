from datetime import datetime


class Stats:
    def __init__(self, command, stats_id):
        self.command = command
        self.response = None
        self.stats_id = stats_id

        self.start_time = datetime.now()
        self.end_time = None
        self.duration = None

    def add_response(self, response):
        self.response = response
        self.end_time = datetime.now()
        self.duration = self.get_duration()
        # self.print_stats()

    def get_duration(self):
        diff = self.end_time - self.start_time

        return diff.total_seconds()

    def print_stats(self):
        print('\nid: %s' % self.stats_id)
        print('command: %s' % self.command)
        print('response: %s' % self.response)
        print('start time: %s' % self.start_time)
        print('end_time: %s' % self.end_time)
        print('duration: %s\n' % self.duration)

    def got_response(self):
        if self.response is None:
            return False
        else:
            return True

    def return_stats(self):
        string = ''
        string += '\nid: %s\n' % self.stats_id
        string += 'command: %s\n' % self.command
        string += 'response: %s\n' % self.response
        string += 'start time: %s\n' % self.start_time
        string += 'end_time: %s\n' % self.end_time
        string += 'duration: %s\n' % self.duration

        return string
