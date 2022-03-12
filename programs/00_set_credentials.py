update_cred = input('Do you want to update your credentials? [Y/N]:\t').strip()[0].upper()

if update_cred=='Y':
    print('\nConfigure your AWS credentials:\n')
    AWS_REGION_NAME = input('AWS_REGION_NAME:\t').strip()
    AWS_ACCESS_KEY_ID = input('AWS_ACCESS_KEY_ID:\t').strip()
    AWS_SECRET_ACCESS_KEY = input('AWS_SECRET_ACCESS_KEY:\t').strip()

    credentials = f'''
    AWS_REGION_NAME={AWS_REGION_NAME}
    AWS_ACCESS_KEY_ID={AWS_ACCESS_KEY_ID}
    AWS_SECRET_ACCESS_KEY={AWS_SECRET_ACCESS_KEY}
    '''

    print('\nYou have updated your credentials\n')

else:
    print('\nYou have selected not to update your credentials\n')