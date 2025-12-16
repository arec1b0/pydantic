This page provides example snippets for creating more complex, custom validators in Pydantic.
Many of these examples are adapted from Pydantic issues and discussions, and are intended to showcase
the flexibility and power of Pydantic's validation system.

## Custom `datetime` Validator via [`Annotated`][typing.Annotated] Metadata

In this example, we'll construct a custom validator, attached to an [`Annotated`][typing.Annotated] type,
that ensures a [`datetime`][datetime.datetime] object adheres to a given timezone constraint.

The custom validator supports string specification of the timezone, and will raise an error if the [`datetime`][datetime.datetime] object does not have the correct timezone.

We use `__get_pydantic_core_schema__` in the validator to customize the schema of the annotated type (in this case, [`datetime`][datetime.datetime]), which allows us to add custom validation logic. Notably, we use a `wrap` validator function so that we can perform operations both before and after the default `pydantic` validation of a [`datetime`][datetime.datetime].

```python
import datetime as dt
from dataclasses import dataclass
from pprint import pprint
from typing import Annotated, Any, Callable, Optional

import pytz
from pydantic_core import CoreSchema, core_schema

from pydantic import (
    GetCoreSchemaHandler,
    PydanticUserError,
    TypeAdapter,
    ValidationError,
)


@dataclass(frozen=True)
class MyDatetimeValidator:
    tz_constraint: Optional[str] = None

    def tz_constraint_validator(
        self,
        value: dt.datetime,
        handler: Callable,  # (1)!
    ):
        """Validate tz_constraint and tz_info."""
        # handle naive datetimes
        if self.tz_constraint is None:
            assert (
                value.tzinfo is None
            ), 'tz_constraint is None, but provided value is tz-aware.'
            return handler(value)

        # validate tz_constraint and tz-aware tzinfo
        if self.tz_constraint not in pytz.all_timezones:
            raise PydanticUserError(
                f'Invalid tz_constraint: {self.tz_constraint}',
                code='unevaluable-type-annotation',
            )
        result = handler(value)  # (2)!
        assert self.tz_constraint == str(
            result.tzinfo
        ), f'Invalid tzinfo: {str(result.tzinfo)}, expected: {self.tz_constraint}'

        return result

    def __get_pydantic_core_schema__(
        self,
        source_type: Any,
        handler: GetCoreSchemaHandler,
    ) -> CoreSchema:
        return core_schema.no_info_wrap_validator_function(
            self.tz_constraint_validator,
            handler(source_type),
        )


LA = 'America/Los_Angeles'
ta = TypeAdapter(Annotated[dt.datetime, MyDatetimeValidator(LA)])
print(
    ta.validate_python(dt.datetime(2023, 1, 1, 0, 0, tzinfo=pytz.timezone(LA)))
)
#> 2023-01-01 00:00:00-07:53

LONDON = 'Europe/London'
try:
    ta.validate_python(
        dt.datetime(2023, 1, 1, 0, 0, tzinfo=pytz.timezone(LONDON))
    )
except ValidationError as ve:
    pprint(ve.errors(), width=100)
    """
    [{'ctx': {'error': AssertionError('Invalid tzinfo: Europe/London, expected: America/Los_Angeles')},
    'input': datetime.datetime(2023, 1, 1, 0, 0, tzinfo=<DstTzInfo 'Europe/London' LMT-1 day, 23:59:00 STD>),
    'loc': (),
    'msg': 'Assertion failed, Invalid tzinfo: Europe/London, expected: America/Los_Angeles',
    'type': 'assertion_error',
    'url': 'https://errors.pydantic.dev/2.8/v/assertion_error'}]
    """
```

1. The `handler` function is what we call to validate the input with standard `pydantic` validation
2. We call the `handler` function to validate the input with standard `pydantic` validation in this wrap validator

We can also enforce UTC offset constraints in a similar way.  Assuming we have a `lower_bound` and an `upper_bound`, we can create a custom validator to ensure our `datetime` has a UTC offset that is inclusive within the boundary we define:

```python
import datetime as dt
from dataclasses import dataclass
from pprint import pprint
from typing import Annotated, Any, Callable

import pytz
from pydantic_core import CoreSchema, core_schema

from pydantic import GetCoreSchemaHandler, TypeAdapter, ValidationError


@dataclass(frozen=True)
class MyDatetimeValidator:
    lower_bound: int
    upper_bound: int

    def validate_tz_bounds(self, value: dt.datetime, handler: Callable):
        """Validate and test bounds"""
        assert value.utcoffset() is not None, 'UTC offset must exist'
        assert self.lower_bound <= self.upper_bound, 'Invalid bounds'

        result = handler(value)

        hours_offset = value.utcoffset().total_seconds() / 3600
        assert (
            self.lower_bound <= hours_offset <= self.upper_bound
        ), 'Value out of bounds'

        return result

    def __get_pydantic_core_schema__(
        self,
        source_type: Any,
        handler: GetCoreSchemaHandler,
    ) -> CoreSchema:
        return core_schema.no_info_wrap_validator_function(
            self.validate_tz_bounds,
            handler(source_type),
        )


LA = 'America/Los_Angeles'  # UTC-7 or UTC-8
ta = TypeAdapter(Annotated[dt.datetime, MyDatetimeValidator(-10, -5)])
print(
    ta.validate_python(dt.datetime(2023, 1, 1, 0, 0, tzinfo=pytz.timezone(LA)))
)
#> 2023-01-01 00:00:00-07:53

LONDON = 'Europe/London'
try:
    print(
        ta.validate_python(
            dt.datetime(2023, 1, 1, 0, 0, tzinfo=pytz.timezone(LONDON))
        )
    )
except ValidationError as e:
    pprint(e.errors(), width=100)
    """
    [{'ctx': {'error': AssertionError('Value out of bounds')},
    'input': datetime.datetime(2023, 1, 1, 0, 0, tzinfo=<DstTzInfo 'Europe/London' LMT-1 day, 23:59:00 STD>),
    'loc': (),
    'msg': 'Assertion failed, Value out of bounds',
    'type': 'assertion_error',
    'url': 'https://errors.pydantic.dev/2.8/v/assertion_error'}]
    """
```

## Validating Nested Model Fields

Here, we demonstrate two ways to validate a field of a nested model, where the validator utilizes data from the parent model.

In this example, we construct a validator that checks that each user's password is not in a list of forbidden passwords specified by the parent model.

One way to do this is to place a custom validator on the outer model:

```python
from typing_extensions import Self

from pydantic import BaseModel, ValidationError, model_validator


class User(BaseModel):
    username: str
    password: str


class Organization(BaseModel):
    forbidden_passwords: list[str]
    users: list[User]

    @model_validator(mode='after')
    def validate_user_passwords(self) -> Self:
        """Check that user password is not in forbidden list. Raise a validation error if a forbidden password is encountered."""
        for user in self.users:
            current_pw = user.password
            if current_pw in self.forbidden_passwords:
                raise ValueError(
                    f'Password {current_pw} is forbidden. Please choose another password for user {user.username}.'
                )
        return self


data = {
    'forbidden_passwords': ['123'],
    'users': [
        {'username': 'Spartacat', 'password': '123'},
        {'username': 'Iceburgh', 'password': '87'},
    ],
}
try:
    org = Organization(**data)
except ValidationError as e:
    print(e)
    """
    1 validation error for Organization
      Value error, Password 123 is forbidden. Please choose another password for user Spartacat. [type=value_error, input_value={'forbidden_passwords': [...gh', 'password': '87'}]}, input_type=dict]
    """
```

Alternatively, a custom validator can be used in the nested model class (`User`), with the forbidden passwords data from the parent model being passed in via validation context.

!!! warning
    The ability to mutate the context within a validator adds a lot of power to nested validation, but can also lead to confusing or hard-to-debug code. Use this approach at your own risk!

```python
from pydantic import BaseModel, ValidationError, ValidationInfo, field_validator


class User(BaseModel):
    username: str
    password: str

    @field_validator('password', mode='after')
    @classmethod
    def validate_user_passwords(
        cls, password: str, info: ValidationInfo
    ) -> str:
        """Check that user password is not in forbidden list."""
        forbidden_passwords = (
            info.context.get('forbidden_passwords', []) if info.context else []
        )
        if password in forbidden_passwords:
            raise ValueError(f'Password {password} is forbidden.')
        return password


class Organization(BaseModel):
    forbidden_passwords: list[str]
    users: list[User]

    @field_validator('forbidden_passwords', mode='after')
    @classmethod
    def add_context(cls, v: list[str], info: ValidationInfo) -> list[str]:
        if info.context is not None:
            info.context.update({'forbidden_passwords': v})
        return v


data = {
    'forbidden_passwords': ['123'],
    'users': [
        {'username': 'Spartacat', 'password': '123'},
        {'username': 'Iceburgh', 'password': '87'},
    ],
}

try:
    org = Organization.model_validate(data, context={})
except ValidationError as e:
    print(e)
    """
    1 validation error for Organization
    users.0.password
      Value error, Password 123 is forbidden. [type=value_error, input_value='123', input_type=str]
    """
```

Note that if the context property is not included in `model_validate`, then `info.context` will be `None` and the forbidden passwords list will not get added to the context in the above implementation. As such, `validate_user_passwords` would not carry out the desired password validation.

More details about validation context can be found in the [validators documentation](../concepts/validators.md#validation-context).

## Cross-Field Validation

Often you need to validate a field based on the value of one or more other fields. This section demonstrates several patterns for implementing cross-field validation.

### Using `model_validator` for Multi-Field Validation

The most straightforward approach for validation that depends on multiple fields is to use a [`model_validator`][pydantic.model_validator] with `mode='after'`. This gives you access to all validated field values:

```python
from decimal import Decimal

from typing_extensions import Self

from pydantic import BaseModel, ValidationError, model_validator


class Order(BaseModel):
    subtotal: Decimal
    discount_percent: Decimal
    tax_percent: Decimal
    total: Decimal

    @model_validator(mode='after')
    def validate_total(self) -> Self:
        """Ensure the total matches the calculated value based on subtotal, discount, and tax."""
        expected_total = self.subtotal * (1 - self.discount_percent / 100) * (
            1 + self.tax_percent / 100
        )
        # Allow for small floating point differences
        if abs(self.total - expected_total) > Decimal('0.01'):
            raise ValueError(
                f'Total {self.total} does not match expected value {expected_total:.2f} '
                f'(subtotal={self.subtotal}, discount={self.discount_percent}%, tax={self.tax_percent}%)'
            )
        return self


# Valid order
order = Order(
    subtotal=Decimal('100.00'),
    discount_percent=Decimal('10'),
    tax_percent=Decimal('8'),
    total=Decimal('97.20'),  # 100 * 0.9 * 1.08 = 97.20
)
print(order)
#> subtotal=Decimal('100.00') discount_percent=Decimal('10') tax_percent=Decimal('8') total=Decimal('97.20')

# Invalid order - wrong total
try:
    Order(
        subtotal=Decimal('100.00'),
        discount_percent=Decimal('10'),
        tax_percent=Decimal('8'),
        total=Decimal('100.00'),  # Wrong!
    )
except ValidationError as e:
    print(e)
    """
    1 validation error for Order
      Value error, Total 100.00 does not match expected value 97.20 (subtotal=100.00, discount=10%, tax=8%) [type=value_error, input_value={'subtotal': Decimal('100....otal': Decimal('100.00')}, input_type=dict]
    """
```

### Date Range Validation

A common use case is validating that a start date comes before an end date:

```python
import datetime as dt

from typing_extensions import Self

from pydantic import BaseModel, ValidationError, model_validator


class DateRange(BaseModel):
    start_date: dt.date
    end_date: dt.date
    min_days: int = 1
    max_days: int | None = None

    @model_validator(mode='after')
    def validate_date_range(self) -> Self:
        """Validate that end_date is after start_date and within allowed range."""
        if self.end_date < self.start_date:
            raise ValueError(
                f'end_date ({self.end_date}) must be after start_date ({self.start_date})'
            )

        duration = (self.end_date - self.start_date).days
        if duration < self.min_days:
            raise ValueError(
                f'Date range must be at least {self.min_days} day(s), got {duration}'
            )

        if self.max_days is not None and duration > self.max_days:
            raise ValueError(
                f'Date range must be at most {self.max_days} day(s), got {duration}'
            )

        return self


# Valid range
event = DateRange(
    start_date=dt.date(2024, 1, 1),
    end_date=dt.date(2024, 1, 7),
    min_days=1,
    max_days=30,
)
print(event)
#> start_date=datetime.date(2024, 1, 1) end_date=datetime.date(2024, 1, 7) min_days=1 max_days=30

# Invalid: end before start
try:
    DateRange(
        start_date=dt.date(2024, 1, 7),
        end_date=dt.date(2024, 1, 1),
    )
except ValidationError as e:
    print(e)
    """
    1 validation error for DateRange
      Value error, end_date (2024-01-01) must be after start_date (2024-01-07) [type=value_error, input_value={'start_date': datetime.da...e.date(2024, 1, 1), ...}, input_type=dict]
    """
```

### Conditional Field Requirements

Sometimes a field should only be required (or have certain constraints) based on another field's value:

```python
from typing_extensions import Self

from pydantic import BaseModel, ValidationError, model_validator


class PaymentMethod(BaseModel):
    payment_type: str  # 'credit_card', 'bank_transfer', 'crypto'
    card_number: str | None = None
    card_cvv: str | None = None
    bank_account: str | None = None
    bank_routing: str | None = None
    wallet_address: str | None = None

    @model_validator(mode='after')
    def validate_payment_fields(self) -> Self:
        """Ensure required fields are present based on payment type."""
        if self.payment_type == 'credit_card':
            if not self.card_number:
                raise ValueError(
                    'card_number is required for credit card payments'
                )
            if not self.card_cvv:
                raise ValueError('card_cvv is required for credit card payments')

        elif self.payment_type == 'bank_transfer':
            if not self.bank_account:
                raise ValueError(
                    'bank_account is required for bank transfer payments'
                )
            if not self.bank_routing:
                raise ValueError(
                    'bank_routing is required for bank transfer payments'
                )

        elif self.payment_type == 'crypto':
            if not self.wallet_address:
                raise ValueError(
                    'wallet_address is required for crypto payments'
                )

        return self


# Valid credit card payment
cc_payment = PaymentMethod(
    payment_type='credit_card',
    card_number='4111111111111111',
    card_cvv='123',
)
print(cc_payment.payment_type)
#> credit_card

# Missing required field for payment type
try:
    PaymentMethod(
        payment_type='credit_card',
        card_number='4111111111111111',
        # Missing card_cvv!
    )
except ValidationError as e:
    print(e)
    """
    1 validation error for PaymentMethod
      Value error, card_cvv is required for credit card payments [type=value_error, input_value={'payment_type': 'credit_c...11111111111111111', ...}, input_type=dict]
    """
```

### Using `field_validator` with `ValidationInfo.data`

For simpler cross-field validation, you can use a [`field_validator`][pydantic.field_validator] that accesses previously validated fields through [`ValidationInfo.data`][pydantic.ValidationInfo.data]:

```python
from pydantic import BaseModel, ValidationError, ValidationInfo, field_validator


class Rectangle(BaseModel):
    width: float
    height: float
    diagonal: float

    @field_validator('diagonal', mode='after')
    @classmethod
    def validate_diagonal(cls, v: float, info: ValidationInfo) -> float:
        """Validate that the diagonal matches the Pythagorean theorem."""
        width = info.data.get('width')
        height = info.data.get('height')
        if width is not None and height is not None:
            expected = (width**2 + height**2) ** 0.5
            if abs(v - expected) > 0.01:
                raise ValueError(
                    f'Diagonal {v} does not match expected value {expected:.2f} '
                    f'for width={width} and height={height}'
                )
        return v


# Valid rectangle
rect = Rectangle(width=3.0, height=4.0, diagonal=5.0)
print(rect)
#> width=3.0 height=4.0 diagonal=5.0

# Invalid diagonal
try:
    Rectangle(width=3.0, height=4.0, diagonal=6.0)
except ValidationError as e:
    print(e)
    """
    1 validation error for Rectangle
    diagonal
      Value error, Diagonal 6.0 does not match expected value 5.00 for width=3.0 and height=4.0 [type=value_error, input_value=6.0, input_type=float]
    """
```

!!! warning
    When using `ValidationInfo.data`, remember that fields are validated in definition order. You can only access fields that are defined **before** the field being validated. In the example above, `diagonal` is defined after `width` and `height`, so both are available in `info.data`.

## Validation with External Calls

!!! warning "Use with Caution"
    Performing external calls (HTTP requests, database queries, file system operations) during validation is **generally not recommended** for several reasons:

    1. **Performance**: Validation happens frequently and synchronously. External calls add latency and can significantly slow down your application.
    2. **Reliability**: External services may be unavailable, causing validation to fail unexpectedly.
    3. **Side Effects**: Validation should ideally be pure and deterministic. External calls introduce side effects and non-determinism.
    4. **Testing Difficulty**: Tests become more complex when validation depends on external state.
    5. **Retry Logic**: Failed external calls during validation are hard to retry gracefully.

    Consider alternatives like:

    - Validating external data **before** constructing your Pydantic models
    - Using a separate validation step **after** model construction
    - Caching external validation results when appropriate
    - Using async validation outside of Pydantic if external calls are truly necessary

That said, there are cases where external validation during model construction is the most pragmatic approach. Here are some examples with appropriate safeguards.

### Validating Against a Database

This example shows how to validate that a referenced ID exists in a database. Note the use of caching to minimize database calls:

```python
from functools import lru_cache

from typing_extensions import Self

from pydantic import BaseModel, ValidationError, model_validator


# Simulated database
_users_db: dict[int, dict] = {
    1: {'id': 1, 'name': 'Alice', 'active': True},
    2: {'id': 2, 'name': 'Bob', 'active': True},
    3: {'id': 3, 'name': 'Charlie', 'active': False},
}


@lru_cache(maxsize=100)
def get_user_from_db(user_id: int) -> dict | None:
    """Fetch user from database with caching to reduce DB calls."""
    # In a real application, this would query your database
    return _users_db.get(user_id)


class OrderAssignment(BaseModel):
    order_id: str
    assigned_user_id: int
    require_active_user: bool = True

    @model_validator(mode='after')
    def validate_user_exists(self) -> Self:
        """Validate that the assigned user exists and is active (if required)."""
        user = get_user_from_db(self.assigned_user_id)

        if user is None:
            raise ValueError(f'User with ID {self.assigned_user_id} not found')

        if self.require_active_user and not user.get('active', False):
            raise ValueError(
                f"User '{user['name']}' (ID: {self.assigned_user_id}) is not active"
            )

        return self


# Valid assignment
assignment = OrderAssignment(order_id='ORD-001', assigned_user_id=1)
print(assignment)
#> order_id='ORD-001' assigned_user_id=1 require_active_user=True

# User doesn't exist
try:
    OrderAssignment(order_id='ORD-002', assigned_user_id=999)
except ValidationError as e:
    print(e)
    """
    1 validation error for OrderAssignment
      Value error, User with ID 999 not found [type=value_error, input_value={'order_id': 'ORD-002', 'assigned_user_id': 999}, input_type=dict]
    """

# User exists but is inactive
try:
    OrderAssignment(order_id='ORD-003', assigned_user_id=3)
except ValidationError as e:
    print(e)
    """
    1 validation error for OrderAssignment
      Value error, User 'Charlie' (ID: 3) is not active [type=value_error, input_value={'order_id': 'ORD-003', 'assigned_user_id': 3}, input_type=dict]
    """
```

### Using Validation Context to Control External Validation

You can use validation context to skip external validation in certain scenarios (e.g., during testing or bulk imports):

```python
from typing_extensions import Self

from pydantic import BaseModel, ValidationError, ValidationInfo, model_validator


# Simulated external service
def check_email_deliverable(email: str) -> bool:
    """Simulate checking if an email address is deliverable."""
    # In reality, this might call an email verification API
    blocked_domains = ['example.com', 'test.com', 'invalid.com']
    domain = email.split('@')[-1] if '@' in email else ''
    return domain not in blocked_domains


class UserRegistration(BaseModel):
    email: str
    username: str

    @model_validator(mode='after')
    def validate_email_deliverable(self) -> Self:
        """Validate email is deliverable unless skip_external_validation is set."""
        # Access validation context with proper null check
        context = getattr(self, '__pydantic_validator__', None)

        # Get context from ValidationInfo if available during validation
        # This is a workaround since model_validator doesn't directly receive info
        return self

    @classmethod
    def create_with_validation(
        cls, data: dict, skip_external: bool = False
    ) -> 'UserRegistration':
        """Factory method that optionally skips external validation."""
        if skip_external:
            # Skip external checks (useful for testing/imports)
            return cls.model_validate(data)

        # Perform external validation before model creation
        email = data.get('email', '')
        if not check_email_deliverable(email):
            raise ValueError(f"Email '{email}' is not deliverable")

        return cls.model_validate(data)


# Valid email
user = UserRegistration.create_with_validation(
    {'email': 'user@gmail.com', 'username': 'testuser'}
)
print(user)
#> email='user@gmail.com' username='testuser'

# Invalid email domain
try:
    UserRegistration.create_with_validation(
        {'email': 'user@example.com', 'username': 'testuser'}
    )
except ValueError as e:
    print(e)
    #> Email 'user@example.com' is not deliverable

# Skip validation for testing
test_user = UserRegistration.create_with_validation(
    {'email': 'user@test.com', 'username': 'testuser'}, skip_external=True
)
print(test_user)
#> email='user@test.com' username='testuser'
```

### File Path Validation

Here's an example that validates file paths exist on the filesystem, with options to control the behavior:

```python
import os
from pathlib import Path
from typing import Annotated, Any

from pydantic_core import CoreSchema, core_schema

from pydantic import GetCoreSchemaHandler, ValidationError, ValidationInfo


class ValidatedPath:
    """A path validator that checks if the path exists."""

    def __init__(
        self,
        must_exist: bool = True,
        must_be_file: bool = False,
        must_be_dir: bool = False,
    ):
        self.must_exist = must_exist
        self.must_be_file = must_be_file
        self.must_be_dir = must_be_dir

    def validate(self, value: Any, info: ValidationInfo) -> Path:
        """Validate the path based on configured requirements."""
        # Check if validation should be skipped via context
        if info.context and info.context.get('skip_path_validation'):
            return Path(value) if not isinstance(value, Path) else value

        path = Path(value) if not isinstance(value, Path) else value

        if self.must_exist and not path.exists():
            raise ValueError(f"Path '{path}' does not exist")

        if self.must_be_file and not path.is_file():
            raise ValueError(f"Path '{path}' is not a file")

        if self.must_be_dir and not path.is_dir():
            raise ValueError(f"Path '{path}' is not a directory")

        return path

    def __get_pydantic_core_schema__(
        self, source_type: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        return core_schema.with_info_after_validator_function(
            self.validate, core_schema.str_schema()
        )


from pydantic import BaseModel, TypeAdapter

# Example usage
ExistingFile = Annotated[Path, ValidatedPath(must_exist=True, must_be_file=True)]
ExistingDir = Annotated[Path, ValidatedPath(must_exist=True, must_be_dir=True)]


class ConfigFile(BaseModel):
    config_path: ExistingFile


# Create a temporary file for testing
import tempfile

with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
    f.write('{}')
    temp_path = f.name

try:
    # Valid - file exists
    config = ConfigFile.model_validate(
        {'config_path': temp_path}, context={'skip_path_validation': False}
    )
    print(f'Config path: {config.config_path.name}')
    #> Config path: tmp...

    # Invalid - file doesn't exist
    try:
        ConfigFile(config_path='/nonexistent/path/config.json')
    except ValidationError as e:
        print('Validation failed:', e.errors()[0]['msg'])
        #> Validation failed: Value error, Path '/nonexistent/path/config.json' does not exist

    # Skip validation via context
    config_skip = ConfigFile.model_validate(
        {'config_path': '/any/path.json'}, context={'skip_path_validation': True}
    )
    print(f'Skipped validation: {config_skip.config_path}')
    #> Skipped validation: /any/path.json
finally:
    os.unlink(temp_path)  # Clean up
```

### Best Practices for External Validation

When you do need external validation, follow these guidelines:

1. **Use Caching**: Cache results of external calls when possible to reduce latency and load on external services.

2. **Provide Skip Mechanisms**: Use validation context or factory methods to allow skipping external validation in tests or specific scenarios.

3. **Handle Failures Gracefully**: Wrap external calls in try/except blocks and provide meaningful error messages.

4. **Consider Timeouts**: If making network calls, always use timeouts to prevent hanging.

5. **Log Validation Failures**: External validation failures often indicate data quality issues worth investigating.

```python
import logging
from functools import lru_cache

from typing_extensions import Self

from pydantic import BaseModel, ValidationError, model_validator

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1000)
def validate_product_sku(sku: str) -> tuple[bool, str]:
    """
    Validate SKU exists in product catalog.
    Returns (is_valid, error_message).
    """
    # Simulated product catalog lookup
    valid_skus = {'SKU-001', 'SKU-002', 'SKU-003'}

    if sku in valid_skus:
        return True, ''
    return False, f"SKU '{sku}' not found in product catalog"


class OrderLineItem(BaseModel):
    sku: str
    quantity: int

    @model_validator(mode='after')
    def validate_sku_exists(self) -> Self:
        """Validate that the SKU exists in the product catalog."""
        try:
            is_valid, error_message = validate_product_sku(self.sku)
            if not is_valid:
                logger.warning(
                    'Invalid SKU in order: %s', self.sku
                )  # Log for monitoring
                raise ValueError(error_message)
        except ConnectionError as e:
            # Decide how to handle external service failures
            logger.error('Failed to validate SKU %s: %s', self.sku, e)
            # Option 1: Fail validation
            raise ValueError(f'Unable to validate SKU: service unavailable') from e
            # Option 2: Allow through with warning (uncomment to use)
            # logger.warning('Skipping SKU validation due to service error')
        return self


# Valid SKU
item = OrderLineItem(sku='SKU-001', quantity=2)
print(item)
#> sku='SKU-001' quantity=2

# Invalid SKU
try:
    OrderLineItem(sku='INVALID', quantity=1)
except ValidationError as e:
    print(e)
    """
    1 validation error for OrderLineItem
      Value error, SKU 'INVALID' not found in product catalog [type=value_error, input_value={'sku': 'INVALID', 'quantity': 1}, input_type=dict]
    """
```
