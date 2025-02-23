#!/bin/bash
set -e
DEFAULT_USER_ID=1000

if [-v USER_ID] && ["$USER_ID" != "$DEFAULT_USER_ID"  ]; then
    echo "change docker user id to mactch to host's user id ($USER_ID) "

    usermod --uid $USER_ID dkhai
    # all files in the home dir are owned by the new user ID
    find /home/dkhai -user $DEFAULT_USER_ID -exec chown -h $USER_ID {} \;
fi
cd /home/dkhai

# In case no command is provided, set bash to start interractive shell
if [ -z "$1"]; then
    set - "/bin/bash" -l
fi

#run the provided command using user docker user
exec "$@"