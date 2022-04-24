
import time
from optparse import OptionParser


def print_usage():
    print("workflow_scaffold.py -c start -y yaml_conf_path -t date_id  #save workflow and run numerous")
    print("workflow_scaffold.py -c kill -y yaml_conf_path  #kill numerous job")
    print("workflow_scaffold.py -c help")


def main():
    parser = OptionParser()
    parser.add_option("-y", "--conf_yaml", dest="conf_yaml", help="conf_yaml")
    parser.add_option("-t", "--date_id", dest="date_id", help="data date_id")
    parser.add_option("-c", "--command", dest="command", help="command")
    (options, args) = parser.parse_args()

    conf_yaml = options.conf_yaml
    command = options.command
    cycle = options.date_id
    print("input: conf_yaml={}; command={}; cycle={}".format(conf_yaml, command, cycle))

    if cycle is None:
        cycle = str(time.strftime("%Y%m%d", time.localtime()))

    print("refine: conf_yaml={}; command={}; cycle={}".format(conf_yaml, command, cycle))
    
    if command == "kill":
        print("kill job")
    elif command=="start":
        print("start job ")
    else:
        print_usage()
    

if __name__ == '__main__':
    main()