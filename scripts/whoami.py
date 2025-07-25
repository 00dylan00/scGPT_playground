import getpass

# Get the username
username = getpass.getuser()

# Print the username
print(username)

# Save the username to a text file
with open('username.txt', 'w') as file:
    file.write(username)