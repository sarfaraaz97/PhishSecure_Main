"""Add created_at column to URLData

Revision ID: ac76349e6c5f
Revises: 
Create Date: 2025-01-25 10:48:38.015218

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'ac76349e6c5f'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table('url_data', schema=None) as batch_op:
        batch_op.add_column(sa.Column('created_at', sa.DateTime(), nullable=True))

    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table('url_data', schema=None) as batch_op:
        batch_op.drop_column('created_at')

    # ### end Alembic commands ###
