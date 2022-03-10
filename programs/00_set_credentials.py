print('\nConfigure your AWS credentials:\n')
AWS_REGION_NAME = input('AWS_REGION_NAME:\t')
AWS_ACCESS_KEY_ID = input('AWS_ACCESS_KEY_ID:\t')
AWS_SECRET_ACCESS_KEY = input('AWS_SECRET_ACCESS_KEY:\t')

credentials = f'''
AWS_REGION_NAME={AWS_REGION_NAME}
AWS_ACCESS_KEY_ID={AWS_ACCESS_KEY_ID}
AWS_SECRET_ACCESS_KEY={AWS_SECRET_ACCESS_KEY}
'''
with open('.env','w') as f:
    f.write(credentials)
