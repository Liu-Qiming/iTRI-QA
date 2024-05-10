### **Administrator Manual**

#### **Creating a New Group**

To set up a new group for project collaboration:

```bash
sudo groupadd new-group-name
```

- **Step 1:** This command creates a new group named `new-group-name` for project collaboration.

#### **Adding a New User and Granting Group Permissions**

To add a new user to the server and assign them to the `group-name` group:

```bash
sudo adduser newusername
sudo usermod -aG group-name newusername
```

- **Step 1:** Replace `newusername` with the actual username for the new user, and follow the instruction.
- **Step 2:** This adds the new user to the `group-name` group, granting them access to the shared project resources.

#### **Managing SSH Service**

To manage the SSH service:

```bash
sudo systemctl status ssh #check status
sudo systemctl restart ssh #restart ssh
sudo systemctl stop ssh #terminate ssh
```

- **Step 1:** These commands allow you to check the status, restart, or stop the SSH service on the server.

#### **Checking Public IP Address**

To check the server's public IP address:

```bash
curl ifconfig.me
```

- **Step 1:** This command uses `curl` to retrieve the public IP from an external service, useful for confirming the server's outward-facing IP address.

#### **Granting Group Access to a Specific Directory**

To set permissions for the `group-name` group on a specific directory:

```bash
sudo chown -R :group-name /srv/directory-you-want-to-share
sudo chmod -R 2775 /srv/directory-you-want-to-share
```

- **Step 1:** These commands change the ownership of the `/srv/directory-you-want-to-share` directory to the `group-name` group and set permissions so that all group members can read, write, and execute files while new files inherit the group from the directory.

- Here are additional instructions to include in the **Administrator Manual** to manage user groups and directory access, which cover removing a user from a group, deleting a group, and revoking a group's access to a directory.

#### **Removing a User from a Group**

To remove a user from a specific group:

```bash
sudo gpasswd -d username group-name
```

- **Step 1:** Replace `username` with the actual username of the user you want to remove from the group.
- **Step 2:** This command removes the user from the `group-name` group.

#### **Deleting a Group**

To completely remove a group from the system:

```bash
sudo groupdel group-name
```

- **Step 1:** This command deletes the `group-name` group. Make sure that no important permissions or access controls are solely dependent on this group before deleting it.

#### **Revoking Group Access to a Directory**

To remove a group's access permissions from a specific directory:

```bash
sudo setfacl -x g:group-name /srv/directory-want-to-limit-access
```

- **Step 1:** This command removes all special access control list (ACL) permissions for the `group-name` group from the `/srv/directory-want-to-limit-access` directory.
If you wish to revert to standard permissions:
 ```bash
 sudo chown -R root:root /srv/directory-want-to-limit-access
 sudo chmod -R 755 /srv/directory-want-to-limit-access
 ```
 - These commands reset the ownership to `root` and change the permissions so that only the owner has write access, while everyone else has read and execute permissions.

## Appendix
### Understanding Unix/Linux File Permissions

In Unix and Linux systems, file permissions control the level of access that users have to files and directories. These permissions determine who can read, write, or execute the files. Here's a quick tutorial on understanding these permissions:

#### Numeric Permissions:

Permissions are represented numerically for convenience, with each digit representing a different class of users:
- The **first digit** is for the user who owns the file (owner).
- The **second digit** is for the group that owns the file (group).
- The **third digit** is for all other users (others).

Each digit can be a number from 0 to 7, derived by adding:
- 4 for read (r) permissions.
- 2 for write (w) permissions.
- 1 for execute (x) permissions.

#### Special Permissions:

Sometimes, there are four digits in permission settings:
- The **fourth digit** (from the left) represents special permissions such as setuid, setgid, or sticky bit.

- **2xxx** (setgid): When set on a directory, new files created within inherit their group from the directory (not from the user who created the file).
- **4xxx** (setuid): When set on an executable file, the file is executed with the permissions of the file owner.
- **1xxx** (sticky bit): Typically used on directories, when set, files can only be renamed or deleted by the file's owner, the directory's owner, or the root user.

### Permissions Table

Here is a table representing combinations of permissions from 0 to 7 for both owner, group, and others:

|    | 0 (*) | 1 (x) | 2 (w) | 3 (wx) | 4 (r) | 5 (rx) | 6 (rw) | 7 (rwx) |
|----|-------|-------|-------|--------|-------|--------|--------|---------|
| 0  |  ---  |  --x  |  -w-  |  -wx   | r--   | r-x    | rw-    | rwx     |
| 1  |  --x  |  --x  |  -wx  |  -wx   | r-x   | r-x    | rwx    | rwx     |
| 2  |  -w-  |  -wx  |  -w-  |  -wx   | rw-   | rwx    | rw-    | rwx     |
| 3  |  -wx  |  -wx  |  -wx  |  -wx   | rwx   | rwx    | rwx    | rwx     |
| 4  | r--   | r-x   | rw-   | rwx    | r--   | r-x    | rw-    | rwx     |
| 5  | r-x   | r-x   | rwx   | rwx    | r-x   | r-x    | rwx    | rwx     |
| 6  | rw-   | rwx   | rw-   | rwx    | rw-   | rwx    | rw-    | rwx     |
| 7  | rwx   | rwx   | rwx   | rwx    | rwx   | rwx    | rwx    | rwx     |

- The leftmost column (0-7) represents the current mode of the owner.
- The top row (0-7) represents the mode applied, for example, by `chmod`.
- Each cell shows the resulting permissions if you apply the mode from the top row to the mode from the leftmost column.

### Examples:

- **chmod 700** on a file sets the permissions to `rwx` for the owner, and no access (`---`) for the group and others.
- **chmod 2775** on a directory sets full permissions (`rwx`) for the owner and the group, with the setgid bit set (making new files inherit the group), and read and execute permissions (`r-x`) for others.
