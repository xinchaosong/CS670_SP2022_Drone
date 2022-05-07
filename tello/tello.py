from datetime import datetime
import os
import socket
import threading
import time

import cv2
import numpy as np

from tello.stats import Stats
import h264decoder


class Tello:
    def __init__(self):
        """
        Initialization.
        """
        self.local_ip = ''
        self.tello_ip = '192.168.10.1'
        self.local_port = 8889
        self.local_video_port = 11111  # port for receiving video stream
        self.tello_port = 8889
        self.tello_address = (self.tello_ip, self.tello_port)
        self.decoder = h264decoder.H264Decoder()
        self.frame = None  # numpy array BGR -- current camera output frame
        self.frame = None  # numpy array BGR -- current camera output frame
        self.response = None
        self.log = []
        self.MAX_TIME_OUT = 15.0
        self.tello_on = True
        self.record_video_on = False
        self.recording_frames = []

        # socket for sending cmd
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind((self.local_ip, self.local_port))

        # thread for receiving cmd ack
        self.receive_thread = threading.Thread(target=self._receive_thread)
        self.receive_thread.daemon = True
        self.receive_thread.start()

        # to receive video -- send cmd: command, streamon
        self.send_command('command')
        self.send_command('streamon')

        # socket for receiving video stream
        self.socket_video = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket_video.bind((self.local_ip, self.local_video_port))

        # thread for receiving video
        self.receive_video_thread = threading.Thread(target=self._receive_video_thread)
        self.receive_video_thread.daemon = True
        self.receive_video_thread.start()

    def close(self):
        """
        End the remote control for Tello
        """

        self.tello_on = False

        self.send_command('land')
        self.send_command('streamoff')
        time.sleep(0.1)

        self.socket_video.close()
        self.socket.close()

    def read(self):
        """
        Return the last frame from camera.
        """

        return self.frame

    def execute_command(self, command):
        if command != '' and command != '\n':
            command = command.rstrip()

            if command.find('delay') != -1:
                sec = float(command.partition('delay')[2])
                print('delay %s' % sec)
                time.sleep(sec)
                pass
            else:
                self.send_command(command)

    def execute_commands(self, commands):
        for command in commands:
            self.execute_command(command)

        log = self.get_log()
        log_filename = "log_" + self._generate_file_name_by_time() + ".txt"
        with open(log_filename, 'w') as out:
            for stat in log:
                stat.print_stats()
                out.write(stat.return_stats())

    def send_command(self, command):
        """
        Send a command to the ip address. Will be blocked until the last command receives an 'OK'.
        If the command fails (either b/c time out or error), will try to resend the command
        :param command: (str) the command to send
        """

        self.log.append(Stats(command, len(self.log)))

        if command.find('snapshot') != -1:
            output_path = command.partition('snapshot')[2]
            if not output_path:
                output_path = "."

            self.take_snapshot(output_path)

        elif command.find('recordon') != -1:
            self.record_video_on = True
            print("Recording on!")

        elif command.find('recordoff') != -1:
            self.record_video_on = False
            print("Recording off!")

        else:
            self.socket.sendto(command.encode('utf-8'), self.tello_address)
            print('sending command: %s to %s' % (command, self.tello_ip))

            start = time.time()
            while not self.log[-1].got_response():
                now = time.time()
                diff = now - start

                if diff > self.MAX_TIME_OUT:
                    print('Max timeout exceeded... command %s' % command)
                    return

            print('Done!!! sent command: %s to %s' % (command, self.tello_ip))

            if command.find('land') != -1 and self.recording_frames:
                self._save_video_recording()

    def take_snapshot(self, output_path="."):
        """
        Save the current frame of the video as a jpg file and put it into output_path
        :param output_path: the output path for the snapshot
        :return: the location of the snapshot saved
        """

        # grab the current timestamp and use it to construct the filename
        filename = self._generate_file_name_by_time() + ".jpg"
        output_path = os.path.sep.join((output_path, filename))

        # save the file
        try:
            cv2.imwrite(output_path, cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB))
            print("[INFO] saved {}".format(filename))
        except Exception:
            print("Failed to save the snapshot.")

        return output_path

    def get_frame(self):
        """
        Get the current frame of the video.
        :return: self.frame
        """

        return self.frame

    def get_recording_frames(self):
        """
        Get the all frames of the recording.
        :return: self.recording_frames
        """

        return self.recording_frames

    def get_log(self):
        """
        Get the log.
        :return:
        """

        return self.log

    def _save_video_recording(self, output_path="."):
        """
        Save all frames of the recording as a mp4 file put it into output_path
        :param output_path: the output path for the recording
        :return: the location of the recording saved
        """

        filename = self._generate_file_name_by_time() + ".mp4"
        output_path = os.path.sep.join((output_path, filename))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        frame_rate = 30
        width = len(self.recording_frames[0][0])
        height = len(self.recording_frames[0])
        out = cv2.VideoWriter(output_path, fourcc, frame_rate, (width, height))

        for frame in self.recording_frames:
            # writing to an image array
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            out.write(frame)

        out.release()

        return output_path

    def _receive_thread(self):
        """
        Listen to responses from the Tello.
        Runs as a thread, sets self.response to whatever the Tello last returned.
        """

        while True:
            try:
                self.response, ip = self.socket.recvfrom(4096)
                print('from %s: %s' % (ip, self.response))

                self.log[-1].add_response(self.response)

            except socket.error as es:
                print("Caught exception socket.error : %s" % es)

            except IndexError as ei:
                pass

    def _receive_video_thread(self):
        """
        Listens for video streaming (raw h264) from the Tello.
        Runs as a thread, sets self.frame to the most recent frame Tello captured.
        """

        packet_data = b""

        while True:
            try:
                res_string, ip = self.socket_video.recvfrom(4096)
                packet_data += res_string

                # end of frame
                if len(res_string) != 1460:
                    for frame in self._h264_decode(packet_data):
                        self.frame = frame
                    packet_data = b""

            except socket.error as exc:
                print("Caught exception socket.error : %s" % exc)

    def _h264_decode(self, packet_data):
        """
        Decode raw h264 format data from Tello
        :param packet_data: raw h264 data array
        :return: a list of decoded frame
        """

        res_frame_list = []
        frames = self.decoder.decode(packet_data)

        for frame_data in frames:
            (frame, w, h, ls) = frame_data

            if frame is not None:
                # print('frame size %i bytes, w %i, h %i, linesize %i' % (len(frame), w, h, ls))
                frame = np.frombuffer(frame, dtype=np.ubyte, count=len(frame))
                frame = frame.reshape((h, ls // 3, 3))
                frame = frame[:, :w, :]
                # At this point `frame` references your usual height x width x rgb channels
                # numpy array of unsigned bytes.

                res_frame_list.append(frame)

        if self.record_video_on:
            self.recording_frames += res_frame_list

        return res_frame_list

    @staticmethod
    def _generate_file_name_by_time():
        """
        Generate a file name using the current time.
        :return: a string for the file name generated
        """

        return "{}".format(datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
